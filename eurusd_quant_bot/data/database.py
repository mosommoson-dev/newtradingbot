"""PostgreSQL / TimescaleDB schema and connection helpers.

The schema is intentionally narrow:
    - ``candles``      : OHLCV bars (hypertable on ``ts``)
    - ``trades``       : closed trades (hypertable on ``entry_time``)
    - ``equity``       : equity curve points (hypertable on ``ts``)
    - ``signals``      : every model/strategy signal we generate
    - ``features``     : engineered features (Parquet is preferred for ML, but
                         the latest features are mirrored here for the dashboard)

The module degrades gracefully when ``psycopg2`` or the database itself is not
available; tests can run without a live Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger

from ..config import get_settings

try:  # pragma: no cover - import-time guard, not exercised in CI
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
    _SQLALCHEMY_AVAILABLE = True
except Exception as exc:
    create_engine = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]
    Engine = object  # type: ignore[misc,assignment]
    _SQLALCHEMY_AVAILABLE = False
    logger.warning("sqlalchemy not installed: {}", exc)


SCHEMA_DDL = (
    # Extensions ---------------------------------------------------------
    "CREATE EXTENSION IF NOT EXISTS timescaledb;",
    # candles ------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS candles (
        ts          TIMESTAMPTZ NOT NULL,
        pair        VARCHAR(10) NOT NULL,
        granularity VARCHAR(8)  NOT NULL,
        open        DOUBLE PRECISION NOT NULL,
        high        DOUBLE PRECISION NOT NULL,
        low         DOUBLE PRECISION NOT NULL,
        close       DOUBLE PRECISION NOT NULL,
        volume      DOUBLE PRECISION,
        PRIMARY KEY (pair, granularity, ts)
    );
    """,
    "SELECT create_hypertable('candles', 'ts', if_not_exists => TRUE);",
    # trades -------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS trades (
        id              BIGSERIAL PRIMARY KEY,
        strategy        VARCHAR(50) NOT NULL,
        pair            VARCHAR(10) NOT NULL,
        direction       VARCHAR(4) NOT NULL,
        entry_time      TIMESTAMPTZ NOT NULL,
        exit_time       TIMESTAMPTZ,
        entry_price     DOUBLE PRECISION NOT NULL,
        exit_price      DOUBLE PRECISION,
        lot_size        DOUBLE PRECISION NOT NULL,
        pnl             DOUBLE PRECISION,
        pnl_pips        DOUBLE PRECISION,
        sl_price        DOUBLE PRECISION,
        tp_price        DOUBLE PRECISION,
        commission      DOUBLE PRECISION,
        slippage_pips   DOUBLE PRECISION,
        features        JSONB,
        model_confidence DOUBLE PRECISION
    );
    """,
    "SELECT create_hypertable('trades', 'entry_time', if_not_exists => TRUE);",
    "CREATE INDEX IF NOT EXISTS trades_strategy_idx ON trades (strategy, entry_time DESC);",
    "CREATE INDEX IF NOT EXISTS trades_open_idx ON trades (exit_time NULLS FIRST, entry_time);",
    # equity curve -------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS equity (
        ts            TIMESTAMPTZ NOT NULL,
        balance       DOUBLE PRECISION NOT NULL,
        equity        DOUBLE PRECISION NOT NULL,
        open_pnl      DOUBLE PRECISION,
        drawdown      DOUBLE PRECISION,
        peak_equity   DOUBLE PRECISION,
        PRIMARY KEY (ts)
    );
    """,
    "SELECT create_hypertable('equity', 'ts', if_not_exists => TRUE);",
    # signals ------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS signals (
        id          BIGSERIAL PRIMARY KEY,
        ts          TIMESTAMPTZ NOT NULL,
        strategy    VARCHAR(50) NOT NULL,
        direction   SMALLINT NOT NULL,
        confidence  DOUBLE PRECISION,
        features    JSONB,
        executed    BOOLEAN DEFAULT FALSE
    );
    """,
    "CREATE INDEX IF NOT EXISTS signals_ts_idx ON signals (ts DESC);",
    # latest feature snapshot -------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS features (
        ts          TIMESTAMPTZ NOT NULL,
        pair        VARCHAR(10) NOT NULL,
        granularity VARCHAR(8) NOT NULL,
        payload     JSONB NOT NULL,
        PRIMARY KEY (pair, granularity, ts)
    );
    """,
)


class Database:
    """Light wrapper around a SQLAlchemy ``Engine``.

    Construction does not connect to the database; the engine connects lazily.
    """

    def __init__(self, url: str | None = None) -> None:
        if not _SQLALCHEMY_AVAILABLE:
            raise RuntimeError(
                "sqlalchemy is not installed; install requirements.txt to use the database layer."
            )
        self._url = url or get_settings().database.url
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            assert create_engine is not None
            self._engine = create_engine(self._url, pool_pre_ping=True, future=True)
        return self._engine

    @contextmanager
    def connect(self) -> Iterator[Engine]:
        """Yield a SQLAlchemy connection, ensuring it is closed."""
        with self.engine.connect() as conn:
            yield conn

    def init_schema(self) -> None:
        """Create extensions, tables, and hypertables if they don't exist."""
        assert text is not None
        with self.engine.begin() as conn:
            for stmt in SCHEMA_DDL:
                try:
                    conn.execute(text(stmt))
                except Exception as exc:
                    # `create_hypertable` is idempotent but raises on a regular
                    # Postgres without TimescaleDB; we log and continue so the
                    # bot can still run on plain Postgres.
                    logger.warning("DDL '{}...' skipped: {}", stmt.strip().split()[0], exc)
        logger.info("Database schema ready at {}", self._url)
