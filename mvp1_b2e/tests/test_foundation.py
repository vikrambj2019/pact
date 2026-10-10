import io
import os
import uuid
from datetime import datetime, timedelta, timezone
import pytest
import pytest_asyncio
from fastapi import HTTPException, UploadFile
from sqlalchemy import select, func, event
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB
from mvp1_b2e.api.models import Base, Workspace, Event, BriefRevision, SourceDocument, Job, JobStatus, JobType
from mvp1_b2e.api.routes.workspaces import create_workspace, confirm_question
from mvp1_b2e.api.routes.sources import upload_source
from mvp1_b2e.api.schemas.workspace import WorkspaceCreate, QuestionConfirmation
from mvp1_b2e.api.storage import LocalStorage
from mvp1_b2e.api.config import settings
from mvp1_b2e.jobs.worker import _claim_job, process_one, HANDLERS

@compiles(JSONB, 'sqlite')
def json_sqlite(type_, compiler, **kw):
    return 'JSON'

@pytest_asyncio.fixture
async def factory(tmp_path):
    url = os.environ.get('PACT_TEST_DATABASE_URL', f'sqlite+aiosqlite:///{tmp_path}/test.db')
    engine = create_async_engine(url)
    if url.startswith('sqlite'):
        @event.listens_for(engine.sync_engine, 'connect')
        def foreign_keys(connection, record):
            connection.execute('PRAGMA foreign_keys=ON')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    if not url.startswith('sqlite'):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

async def workspace(session):
    return await create_workspace(WorkspaceCreate(title='Pilot'), session)

@pytest.mark.asyncio
async def test_creation_has_linked_event(factory):
    async with factory() as db:
        ws = await workspace(db)
        events = (await db.execute(select(Event))).scalars().all()
        assert len(events) == 1 and events[0].workspace_id == ws.id

@pytest.mark.asyncio
async def test_question_revision_rejects_stale_and_preserves_history(factory):
    async with factory() as db:
        ws = await workspace(db)
        await confirm_question(ws.id, QuestionConfirmation(primary_question='Original', expected_revision=0), db)
        first_id = ws.current_revision_id
        with pytest.raises(HTTPException) as error:
            await confirm_question(ws.id, QuestionConfirmation(primary_question='Stale', expected_revision=0), db)
        assert error.value.status_code == 409
        await confirm_question(ws.id, QuestionConfirmation(primary_question='Updated', expected_revision=1), db)
        first = await db.get(BriefRevision, first_id)
        second = await db.get(BriefRevision, ws.current_revision_id)
        assert first.structured_snapshot['question']['text'] == 'Original'
        assert second.parent_revision_id == first_id and second.revision_number == 2

async def upload(db, storage, ws_id, text='hello world', key='retry', file=None):
    return await upload_source(ws_id, file=file, pasted_text=None if file else text,
                               idempotency_key=None, idempotency_header=key, db=db, storage=storage)

@pytest.mark.asyncio
async def test_upload_replay_conflict_and_workspace_namespace(factory, tmp_path):
    storage = LocalStorage(tmp_path / 'uploads')
    async with factory() as db:
        ws = await workspace(db)
        first = await upload(db, storage, ws.id)
        again = await upload(db, storage, ws.id)
        assert first == again
        assert (await db.execute(select(func.count()).select_from(SourceDocument))).scalar_one() == 1
        assert len(list(storage.base.rglob('*.txt'))) == 1
        with pytest.raises(HTTPException) as error:
            await upload(db, storage, ws.id, text='different')
        assert error.value.status_code == 409
        other = await workspace(db)
        assert (await upload(db, storage, other.id)).job_id != first.job_id

@pytest.mark.asyncio
@pytest.mark.parametrize('filename,data', [('bad.exe', b'text'), ('bad.txt', b'\xff'), ('bad.pdf', b'not PDF')])
async def test_upload_rejects_unsupported_input(factory, tmp_path, filename, data):
    async with factory() as db:
        ws = await workspace(db)
        with pytest.raises(HTTPException) as error:
            await upload(db, LocalStorage(tmp_path / 'uploads'), ws.id,
                         file=UploadFile(filename=filename, file=io.BytesIO(data)))
        assert error.value.status_code == 422

@pytest.mark.asyncio
async def test_size_and_cumulative_word_limits(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'max_upload_bytes', 20)
    monkeypatch.setattr(settings, 'max_workspace_words', 3)
    storage = LocalStorage(tmp_path / 'uploads')
    async with factory() as db:
        ws = await workspace(db)
        for file in [None, UploadFile(filename='big.txt', file=io.BytesIO(b'x' * 21))]:
            with pytest.raises(HTTPException) as error:
                await upload(db, storage, ws.id, text='x' * 21, file=file)
            assert error.value.status_code == 422
        await upload(db, storage, ws.id)
        with pytest.raises(HTTPException) as error:
            await upload(db, storage, ws.id, text='another pair', key='second')
        assert error.value.status_code == 422

async def seed_job(db, status=JobStatus.pending, attempt=0):
    ws = await workspace(db)
    job = Job(workspace_id=ws.id, type=JobType.parse_source, status=status,
              idempotency_key=str(uuid.uuid4()), input_payload={}, attempt_count=attempt,
              lease_until=datetime.now(timezone.utc) - timedelta(seconds=10))
    db.add(job)
    await db.commit()
    return job

@pytest.mark.asyncio
async def test_expired_job_reclaimed(factory):
    async with factory() as db:
        job = await seed_job(db, JobStatus.running, 1)
        claimed = await _claim_job(db)
        assert claimed.id == job.id and claimed.attempt_count == 2

@pytest.mark.asyncio
async def test_final_expired_attempt_becomes_failed(factory):
    async with factory() as db:
        job = await seed_job(db, JobStatus.running, settings.max_job_attempts)
        assert await _claim_job(db) is None
        await db.refresh(job)
        assert job.status == JobStatus.failed and job.error_code == 'LeaseExpired'

@pytest.mark.asyncio
async def test_handler_failure_rolls_back_partial_writes(factory, monkeypatch):
    async with factory() as db:
        job = await seed_job(db)
        workspace_id, job_id = job.workspace_id, job.id
    async def failing(job, db):
        ws = await db.get(Workspace, workspace_id)
        ws.title = 'Partial change'
        await db.flush()
        raise RuntimeError('private discussion text')
    monkeypatch.setitem(HANDLERS, JobType.parse_source, failing)
    assert await process_one(factory)
    async with factory() as db:
        assert (await db.get(Workspace, workspace_id)).title == 'Pilot'
        job = await db.get(Job, job_id)
        assert job.status == JobStatus.pending
        assert 'private' not in job.error_detail

@pytest.mark.asyncio
async def test_stub_failure_terminal_and_success_clears_lease(factory, monkeypatch):
    async with factory() as db:
        job_id = (await seed_job(db)).id
    assert await process_one(factory)
    async with factory() as db:
        job = await db.get(Job, job_id)
        assert job.status == JobStatus.failed and job.error_code == 'NotImplementedError'
        second_id = (await seed_job(db)).id
    async def success(job, db): return {'ok': True}
    monkeypatch.setitem(HANDLERS, JobType.parse_source, success)
    assert await process_one(factory)
    async with factory() as db:
        job = await db.get(Job, second_id)
        assert job.status == JobStatus.done and job.lease_until is None

@pytest.mark.asyncio
async def test_late_attempt_cannot_complete_reclaimed_job(factory, monkeypatch):
    async with factory() as db:
        job_id = (await seed_job(db)).id
    async def superseded(job, db):
        async with factory() as other:
            current = await other.get(Job, job.id)
            current.attempt_count += 1
            await other.commit()
        return {'obsolete': True}
    monkeypatch.setitem(HANDLERS, JobType.parse_source, superseded)
    await process_one(factory)
    async with factory() as db:
        job = await db.get(Job, job_id)
        assert job.attempt_count == 2 and job.status == JobStatus.running
        assert job.output_payload is None

@pytest.mark.asyncio
async def test_heartbeat_renews_lease(factory, monkeypatch):
    import asyncio
    monkeypatch.setattr(settings, 'job_lease_seconds', 0.3)
    async with factory() as db:
        job_id = (await seed_job(db)).id
    async def slow(job, db):
        await asyncio.sleep(0.45)
        async with factory() as other:
            current = await other.get(Job, job.id)
            lease = current.lease_until.replace(tzinfo=timezone.utc)
            assert lease > datetime.now(timezone.utc)
            assert await _claim_job(other) is None
        return {'ok': True}
    monkeypatch.setitem(HANDLERS, JobType.parse_source, slow)
    await process_one(factory)
    async with factory() as db:
        job = await db.get(Job, job_id)
        assert job.status == JobStatus.done and job.attempt_count == 1

@pytest.mark.asyncio
async def test_postgres_concurrent_question_updates(factory):
    import asyncio
    if factory.kw['bind'].dialect.name != 'postgresql':
        pytest.skip('Requires PACT_TEST_DATABASE_URL for PostgreSQL row locks')
    async with factory() as db:
        ws_id = (await workspace(db)).id
    async def confirm(text):
        async with factory() as db:
            return await confirm_question(ws_id, QuestionConfirmation(primary_question=text, expected_revision=0), db)
    results = await asyncio.gather(confirm('A'), confirm('B'), return_exceptions=True)
    assert sum(isinstance(result, HTTPException) and result.status_code == 409 for result in results) == 1
    async with factory() as db:
        assert (await db.execute(select(func.count()).select_from(BriefRevision))).scalar_one() == 1

@pytest.mark.asyncio
async def test_postgres_concurrent_uploads_return_same_job(factory, tmp_path):
    import asyncio
    if factory.kw['bind'].dialect.name != 'postgresql':
        pytest.skip('Requires PACT_TEST_DATABASE_URL for PostgreSQL row locks')
    async with factory() as db:
        ws_id = (await workspace(db)).id
    storage = LocalStorage(tmp_path / 'uploads')
    async def submit():
        async with factory() as db:
            return await upload(db, storage, ws_id)
    first, second = await asyncio.gather(submit(), submit())
    assert first == second
    assert len(list(storage.base.rglob('*.txt'))) == 1
