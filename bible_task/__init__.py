import json
import logging
import time

from otree.api import *

from llm_rag.config import CHAT_MODEL, EMBEDDING_MODEL, RAG_TOP_K
from llm_rag.pipeline import answer_question


logger = logging.getLogger(__name__)

doc = """
Two-round Bible-study task comparing a baseline LLM and a RAG system.
The system condition is hidden from participants.
"""


class C(BaseConstants):
    NAME_IN_URL = 'bible_task'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 2
    TOTAL_STUDY_PAGES = 11
    MAX_MESSAGE_LENGTH = 2000
    TRANSLATION_NAME = ('World English Bible Updated, Protestant Edition (WEB)')

    SUGGESTED_QUESTIONS = {
        '2 Kings 5:9–14': (
            'Why was Naaman initially unwilling to follow Elisha’s '
            'instruction, and what is the significance of his eventual '
            'response?'
        ),
        'Romans 12:1–5': (
            'What does Paul mean by offering your bodies as a “living '
            'sacrifice,” and how does this relate to transformation and life '
            'in the Christian community?'
        ),
        '1 Samuel 8:4–9': (
            'Why did the elders ask for a king, and why did God describe '
            'their request as a rejection of him?'
        ),
        '1 Corinthians 8:1–6': (
            'What does Paul mean by “knowledge puffs up, but love builds up” '
            'in the discussion of food offered to idols?'
        ),
    }

    # Replace these placeholders with the final approved passage text before
    # pilot testing. Use the same Bible translation for all study passages.

    PASSAGE_TEXTS = {
        '2 Kings 5:9–14': """
    9 So Naaman came with his horses and with his chariots, and stood at the door of the house of Elisha.

    10 Elisha sent a messenger to him, saying, “Go and wash in the Jordan seven times, and your flesh shall come again to you, and you shall be clean.”

    11 But Naaman was angry, and went away and said, “Behold, I thought, ‘He will surely come out to me, and stand, and call on the name of the LORD his God, and wave his hand over the place, and heal the leper.’

    12 Aren’t Abanah and Pharpar, the rivers of Damascus, better than all the waters of Israel? Couldn’t I wash in them and be clean?” So he turned and went away in a rage.

    13 His servants came near and spoke to him, and said, “My father, if the prophet had asked you do some great thing, wouldn’t you have done it? How much rather then, when he says to you, ‘Wash, and be clean’?”

    14 Then went he down and dipped himself seven times in the Jordan, according to the saying of the man of God; and his flesh was restored like the flesh of a little child, and he was clean.
    """.strip(),

        'Romans 12:1–5': """
    1 Therefore I urge you, brothers, by the mercies of God, to present your bodies a living sacrifice, holy, acceptable to God, which is your spiritual service.

    2 Don’t be conformed to this world, but be transformed by the renewing of your mind, so that you may prove what is the good, well-pleasing, and perfect will of God.

    3 For I say through the grace that was given me, to everyone who is among you, not to think of yourself more highly than you ought to think; but to think reasonably, as God has apportioned to each person a measure of faith.

    4 For even as we have many members in one body, and all the members don’t have the same function,

    5 so we, who are many, are one body in Christ, and individually members of one another,
    """.strip(),

        '1 Samuel 8:4–9': """
    4 Then all the elders of Israel gathered themselves together and came to Samuel to Ramah.

    5 They said to him, “Behold, you are old, and your sons don’t walk in your ways. Now make us a king to judge us like all the nations.”

    6 But the thing displeased Samuel when they said, “Give us a king to judge us.” Samuel prayed to the LORD.

    7 The LORD said to Samuel, “Listen to the voice of the people in all that they tell you; for they have not rejected you, but they have rejected me as the king over them.

    8 According to all the works which they have done since the day that I brought them up out of Egypt even to this day, in that they have forsaken me and served other gods, so they also do to you.

    9 Now therefore, listen to their voice. However, you shall protest solemnly to them, and shall show them the way of the king who will reign over them.”
    """.strip(),

        '1 Corinthians 8:1–6': """
    1 Now concerning things sacrificed to idols: We know that we all have knowledge. Knowledge puffs up, but love builds up.

    2 But if anyone thinks that he knows anything, he doesn’t yet know as he ought to know.

    3 But anyone who loves God is known by him.

    4 Therefore concerning the eating of things sacrificed to idols, we know that no idol is anything in the world, and that there is no other God but one.

    5 For though there are things that are called “gods”, whether in the heavens or on earth—as there are many “gods” and many “lords”—

    6 yet to us there is one God, the Father, of whom are all things, and we for him; and one Lord, Jesus Christ, through whom are all things, and we live through him.
    """.strip(),
    }


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


FAMILIARITY_5 = [
    [1, 'Not at all familiar'],
    [2, 'Slightly familiar'],
    [3, 'Moderately familiar'],
    [4, 'Very familiar'],
    [5, 'Extremely familiar'],
]


LIKERT_7 = [
    [1, 'Strongly disagree'],
    [2, 'Disagree'],
    [3, 'Somewhat disagree'],
    [4, 'Neutral'],
    [5, 'Somewhat agree'],
    [6, 'Agree'],
    [7, 'Strongly agree'],
]


POST_TASK_FIELDS = [
    'use01',
    'use02',
    'use03',
    'ctx01',
    'tru01',
    'tru02',
    'tru03',
    'clr01',
    'sat01',
    'int01',
    'int02',
]


class Player(BasePlayer):
    passage_reference = models.StringField()
    system_label = models.StringField()
    system_condition = models.StringField()
    has_interacted = models.BooleanField(initial=False)

    passage_familiarity = models.IntegerField(
        label='Before today, how familiar were you with this Bible passage?',
        choices=FAMILIARITY_5,
        widget=widgets.RadioSelect,
    )

    use01 = models.IntegerField(
        label='This system was useful for studying this passage.',
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )
    use02 = models.IntegerField(
        label='This system helped me understand the passage.',
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )
    use03 = models.IntegerField(
        label='This system made my study of this passage more effective.',
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )
    ctx01 = models.IntegerField(
        label=(
            'This system helped me identify relevant context for '
            'understanding the passage.'
        ),
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )
    tru01 = models.IntegerField(
        label="I could trust the system's answers for this task.",
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )
    tru02 = models.IntegerField(
        label="I felt confident in the system's answers.",
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )
    tru03 = models.IntegerField(
        label="The system's answers were dependable.",
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )
    clr01 = models.IntegerField(
        label="The system's answers were easy to understand.",
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )
    sat01 = models.IntegerField(
        label='Overall, I was satisfied with my experience using this system.',
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )
    int01 = models.IntegerField(
        label='I would use this system again for Bible study.',
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )
    int02 = models.IntegerField(
        label=(
            'If this system were available, I would choose it when I needed '
            'help understanding a Bible passage.'
        ),
        choices=LIKERT_7,
        widget=widgets.RadioSelectHorizontal,
    )


class ChatTurn(ExtraModel):
    player = models.Link(Player)
    turn_index = models.IntegerField()
    question = models.LongStringField()
    answer = models.LongStringField()
    requested_at = models.FloatField()
    completed_at = models.FloatField()
    latency_ms = models.IntegerField()
    passage_reference = models.StringField()
    system_condition = models.StringField()
    status = models.StringField()
    error_type = models.StringField()
    retrieved_sources_json = models.LongStringField()


def creating_session(subsession):
    for player in subsession.get_players():
        participant = player.participant

        if player.round_number == 1:
            player.passage_reference = participant.task_1_passage
            player.system_condition = participant.task_1_condition
            player.system_label = 'Bible Study Assistant A'
        else:
            player.passage_reference = participant.task_2_passage
            player.system_condition = participant.task_2_condition
            player.system_label = 'Bible Study Assistant B'


def participant_is_eligible(player):
    return player.participant.vars.get('eligible_for_study', False)


def progress_context(player, page_offset):
    current_page = 4 + ((player.round_number - 1) * 3) + page_offset
    return dict(
        progress_current=current_page,
        progress_total=C.TOTAL_STUDY_PAGES,
        progress_percent=round(current_page / C.TOTAL_STUDY_PAGES * 100),
    )


def has_interacted_error_message(player, value):
    if not value:
        return 'Please ask the system at least one question before continuing.'


def public_source_summaries(source_records):
    """Return only the source metadata that participants may see."""
    return [
        dict(
            rank=source.get('rank'),
            title=source.get('title', ''),
            source_name=source.get('source_name', ''),
            biblical_reference=source.get('biblical_reference', ''),
            url=source.get('url', ''),
        )
        for source in source_records
    ]


def source_summaries_from_json(value):
    try:
        source_records = json.loads(value or '[]')
    except (TypeError, json.JSONDecodeError):
        logger.warning('A stored retrieval-source record was not valid JSON.')
        return []

    if not isinstance(source_records, list):
        return []
    return public_source_summaries(source_records)


def get_chat_history(player):
    turns = sorted(
        ChatTurn.filter(player=player),
        key=lambda turn: turn.turn_index,
    )
    messages = []
    for turn in turns:
        if turn.status != 'ok':
            continue
        messages.extend(
            [
                dict(sender='participant', text=turn.question),
                dict(
                    sender='system',
                    text=turn.answer,
                    sources=source_summaries_from_json(
                        turn.retrieved_sources_json
                    ),
                ),
            ]
        )
    return messages

def get_model_history(player):
    """Return successful earlier messages from the current task."""
    turns = sorted(
        [
            turn
            for turn in ChatTurn.filter(player=player)
            if turn.status == 'ok' and turn.answer
        ],
        key=lambda turn: turn.turn_index,
    )

    history = []

    for turn in turns:
        history.append(
            dict(role='user', text=turn.question)
        )
        history.append(
            dict(role='model', text=turn.answer)
        )

    return history

class PassageFamiliarity(Page):
    form_model = 'player'
    form_fields = ['passage_familiarity']
    preserve_unsubmitted_inputs = True

    @staticmethod
    def is_displayed(player):
        return participant_is_eligible(player)

    @staticmethod
    def vars_for_template(player):
        context = progress_context(player, 0)
        context.update(
            task_number=player.round_number,
            passage_reference=player.passage_reference,
            passage_text=C.PASSAGE_TEXTS.get(
                player.passage_reference,
                'The passage text is currently unavailable.',
            ),
            translation_name=C.TRANSLATION_NAME,
            familiarity_options=[
                dict(value=value, label=label)
                for value, label in FAMILIARITY_5
            ],
        )
        return context


class BibleTask(Page):
    form_model = 'player'
    form_fields = ['has_interacted']

    @staticmethod
    def is_displayed(player):
        return participant_is_eligible(player)

    @staticmethod
    def vars_for_template(player):
        context = progress_context(player, 1)
        context.update(
            task_number=player.round_number,
            total_tasks=C.NUM_ROUNDS,
            system_label=player.system_label,
            passage_reference=player.passage_reference,
            passage_text=C.PASSAGE_TEXTS.get(
                player.passage_reference,
                'The passage text is currently unavailable.',
            ),
            translation_name=C.TRANSLATION_NAME,
            suggested_question=C.SUGGESTED_QUESTIONS.get(
                player.passage_reference,
                'What is the main message of this passage?',
            ),
        )
        return context

    @staticmethod
    def live_method(player, data):
        if not isinstance(data, dict):
            return

        message_type = data.get('type')

        if message_type == 'load':
            return {
                player.id_in_group: dict(
                    type='history',
                    messages=get_chat_history(player),
                    has_interacted=player.has_interacted,
                )
            }

        if message_type != 'send':
            return

        participant_question = str(data.get('text', '')).strip()

        if not participant_question:
            return {
                player.id_in_group: dict(
                    type='error',
                    message='Please enter a question.',
                )
            }

        if len(participant_question) > C.MAX_MESSAGE_LENGTH:
            return {
                player.id_in_group: dict(
                    type='error',
                    message=(
                        f'Your question must be no longer than '
                        f'{C.MAX_MESSAGE_LENGTH} characters.'
                    ),
                )
            }

        existing_turns = ChatTurn.filter(player=player)
        next_turn_index = len(existing_turns) + 1
        requested_at = time.time()
        passage_text = C.PASSAGE_TEXTS.get(player.passage_reference, '')

        if not passage_text:
            return {
                player.id_in_group: dict(
                    type='error',
                    message='The passage text is currently unavailable.',
                )
            }
        
        conversation_history = get_model_history(player)

        try:
            result = answer_question(
                condition=player.system_condition,
                question=participant_question,
                passage_reference=player.passage_reference,
                passage_text=passage_text,
                conversation_history=conversation_history,
        )
            
        except Exception as error:
            completed_at = time.time()
            logger.exception(
                'Bible-study response failed for participant %s in round %s.',
                player.participant.code,
                player.round_number,
            )
            ChatTurn.create(
                player=player,
                turn_index=next_turn_index,
                question=participant_question,
                answer='',
                requested_at=requested_at,
                completed_at=completed_at,
                latency_ms=round((completed_at - requested_at) * 1000),
                passage_reference=player.passage_reference,
                system_condition=player.system_condition,
                status='error',
                error_type=type(error).__name__,
                retrieved_sources_json='[]',
            )
            return {
                player.id_in_group: dict(
                    type='error',
                    message=(
                        'The system could not generate a response. '
                        'Please try again.'
                    ),
                )
            }

        completed_at = time.time()
        source_records = [source.log_record() for source in result.sources]
        source_log_json = result.source_log_json()

        ChatTurn.create(
            player=player,
            turn_index=next_turn_index,
            question=participant_question,
            answer=result.answer,
            requested_at=requested_at,
            completed_at=completed_at,
            latency_ms=result.latency_ms,
            passage_reference=player.passage_reference,
            system_condition=result.condition,
            status='ok',
            error_type='',
            retrieved_sources_json=source_log_json,
        )

        player.has_interacted = True

        return {
            player.id_in_group: dict(
                type='new_messages',
                messages=[
                    dict(sender='participant', text=participant_question),
                    dict(
                        sender='system',
                        text=result.answer,
                        sources=public_source_summaries(source_records),
                    ),
                ],
                has_interacted=True,
            )
        }


class PostTaskEvaluation(Page):
    form_model = 'player'
    form_fields = POST_TASK_FIELDS
    preserve_unsubmitted_inputs = True

    @staticmethod
    def is_displayed(player):
        return participant_is_eligible(player)

    @staticmethod
    def vars_for_template(player):
        context = progress_context(player, 2)
        context.update(
            task_number=player.round_number,
            system_label=player.system_label,
            next_button_label=(
                'Begin Task 2'
                if player.round_number == 1
                else 'Continue'
            ),
            likert_options=[
                dict(value=value, label=label)
                for value, label in LIKERT_7
            ],
        )
        return context


def custom_export(players):
    yield [
        'study_id',
        'sequence_id',
        'round_number',
        'system_label',
        'system_condition',
        'passage_reference',
        'turn_index',
        'question',
        'answer',
        'requested_at',
        'completed_at',
        'latency_ms',
        'status',
        'error_type',
        'retrieved_sources_json',
        'chat_model',
        'embedding_model',
        'rag_top_k',
    ]

    for player in players:
        turns = sorted(
            ChatTurn.filter(player=player),
            key=lambda turn: turn.turn_index,
        )
        for turn in turns:
            yield [
                player.participant.study_id,
                player.participant.sequence_id,
                player.round_number,
                player.system_label,
                turn.system_condition,
                turn.passage_reference,
                turn.turn_index,
                turn.question,
                turn.answer,
                turn.requested_at,
                turn.completed_at,
                turn.latency_ms,
                turn.status,
                turn.error_type,
                turn.retrieved_sources_json,
                CHAT_MODEL,
                EMBEDDING_MODEL,
                RAG_TOP_K,
            ]


page_sequence = [
    PassageFamiliarity,
    BibleTask,
    PostTaskEvaluation,
]
