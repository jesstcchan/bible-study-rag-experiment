from os import environ

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
    'study_id',
    'sequence_id',
    'passage_block',
    'task_1_passage',
    'task_1_condition',
    'task_2_passage',
    'task_2_condition',
]

SESSION_FIELDS = []

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

SECRET_KEY = '6556891897781'