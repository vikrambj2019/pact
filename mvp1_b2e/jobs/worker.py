"""Job worker — polls the jobs table, claims work with a lease, runs handlers, retries on failure."""
import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, update, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from ..api.config import settings
from ..api.models import Job, JobStatus, JobType

logger = logging.getLogger(__name__)


async def _claim_job(session: AsyncSession) -> Job | None:
    """Atomically claim one pending job with a lease. Returns None if nothing is available."""
    now = datetime.now(timezone.utc)
    lease_until = now + timedelta(seconds=settings.job_lease_seconds)
    # A worker can die after claiming its final attempt. Mark those abandoned
    # jobs terminal rather than leaving them running forever.
    await session.execute(update(Job).where(
        Job.status == JobStatus.running, Job.lease_until <= now,
        Job.attempt_count >= settings.max_job_attempts,
    ).values(status=JobStatus.failed, lease_until=None,
             error_code="LeaseExpired", error_detail="Worker lease expired on final attempt")
        .execution_options(synchronize_session=False))
    result = await session.execute(
        select(Job)
        .where(
            or_(Job.status == JobStatus.pending,
                and_(Job.status == JobStatus.running, Job.lease_until <= now)),
            Job.attempt_count < settings.max_job_attempts,
        )
        .with_for_update(skip_locked=True)
        .order_by(Job.created_at)
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if job is None:
        await session.commit()
        return None
    job.status = JobStatus.running
    job.attempt_count += 1
    job.lease_until = lease_until
    await session.commit()
    await session.refresh(job)
    return job


async def _handle_parse_source(job: Job, session: AsyncSession) -> dict:
    from ..worker.parser import parse_source
    return await parse_source(job.input_payload, session)


HANDLERS = {
    JobType.parse_source: _handle_parse_source,
}


async def _run_job(job: Job, session: AsyncSession):
    handler = HANDLERS.get(job.type)
    if handler is None:
        raise NotImplementedError(f"No handler for job type: {job.type}")
    return await handler(job, session)


async def _heartbeat(factory, job_id, attempt):
    while True:
        await asyncio.sleep(max(0.1, settings.job_lease_seconds / 3))
        async with factory() as session:
            await session.execute(update(Job).where(
                Job.id == job_id, Job.status == JobStatus.running,
                Job.attempt_count == attempt,
            ).values(lease_until=datetime.now(timezone.utc) + timedelta(seconds=settings.job_lease_seconds)))
            await session.commit()


async def process_one(session_factory: async_sessionmaker) -> bool:
    """Claim, execute transactionally, and fence results from superseded attempts."""
    async with session_factory() as session:
        job = await _claim_job(session)
        if job is None:
            return False
        job_id, attempt = job.id, job.attempt_count
    heartbeat = asyncio.create_task(_heartbeat(session_factory, job_id, attempt))
    try:
        async with session_factory() as session:
            job = await session.get(Job, job_id)
            error = None
            try:
                output = await _run_job(job, session)
            except Exception as exc:
                # Roll back all handler writes before recording the failure.
                await session.rollback()
                error = exc
                logger.warning("Job %s failed (%s)", job_id, type(exc).__name__)
            with session.no_autoflush:
                current = (await session.execute(select(Job).where(Job.id == job_id)
                    .with_for_update().execution_options(populate_existing=True))).scalar_one()
            if current.status != JobStatus.running or current.attempt_count != attempt:
                await session.rollback()
                return True
            current.lease_until = None
            if error is None:
                current.status = JobStatus.done
                current.output_payload = output
                current.error_code = current.error_detail = None
            else:
                permanent = isinstance(error, (NotImplementedError, ValueError))
                current.status = JobStatus.failed if permanent or attempt >= settings.max_job_attempts else JobStatus.pending
                current.error_code = type(error).__name__
                # Do not expose source text or credentials embedded in exceptions.
                current.error_detail = "Handler failed; see error_code"
            await session.commit()
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
    return True


async def run_worker():
    """Poll continuously until interrupted."""
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    logger.info("Worker started, polling every %ds", settings.job_poll_interval_seconds)
    while True:
        try:
            did_work = await process_one(factory)
            if not did_work:
                await asyncio.sleep(settings.job_poll_interval_seconds)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Worker loop error")
            await asyncio.sleep(settings.job_poll_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())
