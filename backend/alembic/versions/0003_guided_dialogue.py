"""Persist guided dialogue state and event-idempotency receipts.

Revision ID: 0003_guided_dialogue
Revises: 0002_model_evidence
Create Date: 2026-10-12

The legacy upload/feedback/cities API remains unchanged. Separate dialogue and
event tables keep guided state additive rather than widening ``geo_session``;
the event ledger is the durable idempotency/receipt boundary, while the
dialogue row holds revisioned contract state. Both tables are created only
after the safe unversioned-database adoption flow reaches the existing
baseline, so legacy data is neither inspected nor rewritten here.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_guided_dialogue"
down_revision: Union[str, None] = "0002_model_evidence"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "guided_dialogue",
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("geo_session.id"),
            primary_key=True,
        ),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stage", sa.String(length=20), nullable=False, server_default="country"),
        sa.Column("state_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "guided_event",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("geo_session.id"),
            nullable=False,
        ),
        sa.Column("event_id", sa.String(length=200), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("receipt_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "session_id", "event_id", name="uq_guided_event_session_event"
        ),
    )
    op.create_index("ix_guided_event_session_id", "guided_event", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_guided_event_session_id", table_name="guided_event")
    op.drop_table("guided_event")
    op.drop_table("guided_dialogue")
