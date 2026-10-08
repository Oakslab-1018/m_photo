"""Persist the full model ranking separately from descriptive clues."""
from alembic import op
import sqlalchemy as sa

revision = "0002_model_evidence"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("geo_session", sa.Column("model_evidence_json", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("geo_session", "model_evidence_json")
