from os import environ
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / '.env')

SESSION_CONFIGS = [
    dict(
        name='bible_rag_study',
        display_name='Bible Study: Baseline LLM vs RAG',
        num_demo_participants=1,
        app_sequence=[
            'intro',
            'bible_task',
            'final_survey',
        ],
    ),
]

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00, participation_fee=0.00, doc=""
)

PARTICIPANT_FIELDS = [
    # Random analysis code only; never connect it to a name, email address,
    # recruitment list, participant label, or other identity record.
    'response_code',
    'completed_study',
    'sequence_id',
    'passage_block',
    'task_1_passage',
    'task_1_condition',
    'task_2_passage',
    'task_2_condition',
]

SESSION_FIELDS = []

# Use this room-wide URL without a participant label file. Do not append a
# ``participant_label`` query parameter to recruitment links.
ROOMS = [
    dict(
        name='anonymous_bible_study',
        display_name='Anonymous Bible Study',
    ),
]

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')

DEMO_PAGE_INTRO_HTML = """ """

DEVELOPMENT_SECRET_KEY = 'development-only-change-before-deployment'
SECRET_KEY = environ.get('OTREE_SECRET_KEY', DEVELOPMENT_SECRET_KEY)

if environ.get('OTREE_PRODUCTION') and SECRET_KEY == DEVELOPMENT_SECRET_KEY:
    raise RuntimeError(
        'OTREE_SECRET_KEY must be set to a long random value in production.'
    )
