"""Job worker — polls the jobs table, claims work with a lease, runs handlers, retries on failure."""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from ..api.config import settings
from ..api.models import Job, JobStatus, JobType

logger = logging.getLogger(__name__)


async def _claim_job(session: AsyncSession) -> Job | None:
    """Atomically claim one pending job with a lease. Returns None if nothing is available."""
    now = datetime.now(timezone.utc)
    lease_until = now + timedelta(seconds=settings.job_lease_seconds)
    result = await session.execute(
        select(Job)
        .where(
            Job.status == JobStatus.pending,
            Job.attempt_count < settings.max_job_attempts,
        )
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if job is None:
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


async def process_one(session_factory: async_sessionmaker) -> bool:
    """Try to claim and run one job. Returns True if a job was processed."""
    async with session_factory() as session:
        job = await _claim_job(session)
        if job is None:
            return False
        job_id = job.id
        logger.info("Running job %s type=%s attempt=%d", job_id, job.type, job.attempt_count)

    async with session_factory() as session:
        job = await session.get(Job, job_id)
        try:
            output = await _run_job(job, session)
            job.status = JobStatus.done
            job.output_payload = output
            job.error_code = None
            job.error_detail = None
        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            exhausted = job.attempt_count >= settings.max_job_attempts
            job.status = JobStatus.failed if exhausted else JobStatus.pending
            job.error_code = type(exc).__name__
            job.error_detail = str(exc)[:1000]
            job.lease_until = None
        await session.commit()
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
