from otree.api import *
import random


doc = """
Opening information, consent, eligibility confirmation, and background
questionnaire for the Bible-study LLM and RAG experiment.
"""


class C(BaseConstants):
    NAME_IN_URL = 'intro'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1
    TOTAL_STUDY_PAGES = 11

    # Replace every bracketed value before pilot testing or recruitment.
    STUDY_TITLE = 'AI-Assisted Bible Study: Comparing Two Chat Systems'
    RESEARCHER_NAME = 'Tsz Ching Chan'
    RESEARCHER_AFFILIATION = 'Technical University of Munich'
    RESEARCHER_EMAIL = 'tszching.chan@tum.de'
    SUPPORT_RESOURCE_LABEL = ''
    SUPPORT_RESOURCE_URL = ''

    SEQUENCES = {
        1: ('A', '2 Kings 5:9–14', 'baseline', 'Romans 12:1–5', 'rag'),
        2: ('A', '2 Kings 5:9–14', 'rag', 'Romans 12:1–5', 'baseline'),
        3: ('A', 'Romans 12:1–5', 'baseline', '2 Kings 5:9–14', 'rag'),
        4: ('A', 'Romans 12:1–5', 'rag', '2 Kings 5:9–14', 'baseline'),
        5: ('B', '1 Samuel 8:4–9', 'baseline', '1 Corinthians 8:1–6', 'rag'),
        6: ('B', '1 Samuel 8:4–9', 'rag', '1 Corinthians 8:1–6', 'baseline'),
        7: ('B', '1 Corinthians 8:1–6', 'baseline', '1 Samuel 8:4–9', 'rag'),
        8: ('B', '1 Corinthians 8:1–6', 'rag', '1 Samuel 8:4–9', 'baseline'),
    }


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    consent = models.BooleanField(
        label='Do you voluntarily consent to participate in this study?'
    )

    eligibility_confirmed = models.BooleanField(
        label=(
            'I confirm that I am 18 years of age or older, identify as '
            'Christian, and am comfortable using English for Bible study.'
        )
    )

    age_group = models.StringField(
        label='What is your age group?',
        choices=[
            ['18_24', '18–24 years'],
            ['25_34', '25–34 years'],
            ['35_44', '35–44 years'],
            ['45_54', '45–54 years'],
            ['55_64', '55–64 years'],
            ['65_plus', '65+ years'],
        ],
        widget=widgets.RadioSelect,
    )

    english_level = models.StringField(
        label='How would you describe your current English proficiency?',
        choices=[
            ['A1', 'A1 – Beginner'],
            ['A2', 'A2 – Elementary'],
            ['B1', 'B1 – Intermediate'],
            ['B2', 'B2 – Upper-intermediate'],
            ['C1', 'C1 – Advanced'],
            ['C2', 'C2 – Proficient'],
            ['native_bilingual', 'Native or bilingual'],
            ['unsure', 'Not sure'],
        ],
        widget=widgets.RadioSelect,
    )

    christian_tradition = models.StringField(
        label=(
            'Which broad Christian tradition best describes your '
            'background?'
        ),
        choices=[
            ['catholic', 'Catholic'],
            ['orthodox', 'Orthodox'],
            [
                'protestant',
                (
                    'Protestant'
                ),
            ],
            ['other', 'Other Christian tradition'],
            ['unsure', 'Not sure'],
            ['prefer_not', 'Prefer not to say'],
        ],
        widget=widgets.RadioSelect,
    )

    bible_study_frequency = models.StringField(
        label='How often do you normally engage in personal or group Bible study?',
        choices=[
            ['never', 'Never'],
            ['less_monthly', 'Less than monthly'],
            ['monthly', 'Monthly'],
            ['weekly', 'Weekly'],
            ['several_weekly', 'Several times per week'],
            ['daily', 'Daily'],
        ],
        widget=widgets.RadioSelect,
    )

    ai_use_frequency = models.StringField(
        label='How often have you used AI chatbots such as ChatGPT?',
        choices=[
            ['never', 'Never'],
            ['rarely', 'Rarely'],
            ['sometimes', 'Sometimes'],
            ['often', 'Often'],
            ['very_frequently', 'Very frequently'],
        ],
        widget=widgets.RadioSelect,
    )

    bible_ai_experience = models.BooleanField(
        label=(
            'Have you previously used an AI chatbot for Bible study '
            'or theological questions?'
        ),
        choices=[[True, 'Yes'], [False, 'No']],
        widget=widgets.RadioSelect,
    )

    theological_training = models.StringField(
        label=(
            'What level of formal biblical or theological training '
            'have you completed?'
        ),
        choices=[
            ['none', 'None'],
            ['short_courses', 'Short courses'],
            ['certificate', 'Certificate or diploma'],
            ['undergraduate', 'Undergraduate degree'],
            ['postgraduate', 'Postgraduate degree'],
            ['other', 'Other'],
        ],
        widget=widgets.RadioSelect,
    )


def creating_session(subsession):
    players = subsession.get_players()
    sequence_ids = []

    while len(sequence_ids) < len(players):
        randomized_block = list(C.SEQUENCES.keys())
        random.shuffle(randomized_block)
        sequence_ids.extend(randomized_block)

    for player, sequence_id in zip(players, sequence_ids):
        participant = player.participant
        participant.study_id = participant.code
        participant.sequence_id = sequence_id
        participant.vars['eligible_for_study'] = False
        participant.vars['study_title'] = C.STUDY_TITLE
        participant.vars['researcher_name'] = C.RESEARCHER_NAME
        participant.vars['researcher_affiliation'] = C.RESEARCHER_AFFILIATION
        participant.vars['researcher_email'] = C.RESEARCHER_EMAIL
        participant.vars['support_resource_label'] = C.SUPPORT_RESOURCE_LABEL
        participant.vars['support_resource_url'] = C.SUPPORT_RESOURCE_URL

        (
            passage_block,
            task_1_passage,
            task_1_condition,
            task_2_passage,
            task_2_condition,
        ) = C.SEQUENCES[sequence_id]

        participant.passage_block = passage_block
        participant.task_1_passage = task_1_passage
        participant.task_1_condition = task_1_condition
        participant.task_2_passage = task_2_passage
        participant.task_2_condition = task_2_condition


def progress_context(current_page):
    return dict(
        progress_current=current_page,
        progress_total=C.TOTAL_STUDY_PAGES,
        progress_percent=round(current_page / C.TOTAL_STUDY_PAGES * 100),
    )


def consent_error_message(player, value):
    if value is not True:
        return 'You must consent before continuing.'


def eligibility_confirmed_error_message(player, value):
    if value is not True:
        return 'You must meet and confirm all three eligibility requirements.'


class Welcome(Page):
    form_model = 'player'
    form_fields = ['eligibility_confirmed', 'consent']
    preserve_unsubmitted_inputs = True

    @staticmethod
    def vars_for_template(player):
        context = progress_context(1)
        context.update(
            study_title=C.STUDY_TITLE,
            researcher_name=C.RESEARCHER_NAME,
            researcher_affiliation=C.RESEARCHER_AFFILIATION,
            researcher_email=C.RESEARCHER_EMAIL,
        )
        return context

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.participant.vars['eligible_for_study'] = bool(
            player.consent and player.eligibility_confirmed
        )


class Background(Page):
    form_model = 'player'
    form_fields = [
        'age_group',
        'english_level',
        'christian_tradition',
        'bible_study_frequency',
        'ai_use_frequency',
        'bible_ai_experience',
        'theological_training',
    ]
    preserve_unsubmitted_inputs = True

    @staticmethod
    def is_displayed(player):
        return player.participant.vars.get('eligible_for_study', False)

    @staticmethod
    def vars_for_template(player):
        return progress_context(2)


class Instructions(Page):
    @staticmethod
    def is_displayed(player):
        return player.participant.vars.get('eligible_for_study', False)

    @staticmethod
    def vars_for_template(player):
        return progress_context(3)


page_sequence = [Welcome, Background, Instructions]
