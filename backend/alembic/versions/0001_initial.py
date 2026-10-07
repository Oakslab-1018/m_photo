"""baseline: geo_session + chat_message

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-07

This is the initial schema baseline and mirrors the SQLAlchemy models in
``backend/app/database.py``. It intentionally does NOT touch the legacy
Flask ``database.db``.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "geo_session",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("image_path", sa.String(length=500), nullable=False),
        sa.Column("image_filename", sa.String(length=255), nullable=False),
        sa.Column("clues_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("exclusions_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("confirmed_iso", sa.String(length=10), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "chat_message",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("geo_session.id"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("meta_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_chat_message_session_id", "chat_message", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_message_session_id", table_name="chat_message")
    op.drop_table("chat_message")
    op.drop_table("geo_session")
