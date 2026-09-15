from otree.api import *

from privacy import privacy_error


doc = """
Final comparison questionnaire for the two Bible-study systems.
"""


class C(BaseConstants):
    NAME_IN_URL = 'final_survey'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1
    TOTAL_STUDY_PAGES = 11


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    pref01 = models.StringField(
        label='Overall, which system did you prefer?',
        choices=[
            ['A', 'Bible Study Assistant A'],
            ['B', 'Bible Study Assistant B'],
            ['no_preference', 'No preference'],
        ],
        widget=widgets.RadioSelect,
    )

    # Comma-separated values written by the checkbox interface.
    pref03 = models.StringField(
        label='Which factors most influenced your preference?'
    )

    pref03_other = models.StringField(
        label='If you selected Other, please specify.',
        blank=True,
    )

    pref04 = models.LongStringField(
        label='Please briefly explain your preference.',
        blank=True,
    )

    fb01 = models.LongStringField(
        label='Is there anything else you would like to tell us about the two systems?',
        blank=True,
    )


class FinalComparison(Page):
    form_model = 'player'
    form_fields = ['pref01', 'pref03', 'pref03_other', 'pref04', 'fb01']
    preserve_unsubmitted_inputs = True

    @staticmethod
    def is_displayed(player):
        return player.participant.vars.get('eligible_for_study', False)

    @staticmethod
    def vars_for_template(player):
        return dict(
            progress_current=10,
            progress_total=C.TOTAL_STUDY_PAGES,
            progress_percent=round(10 / C.TOTAL_STUDY_PAGES * 100),
        )

    @staticmethod
    def error_message(player, values):
        selected = {
            value
            for value in str(values.get('pref03') or '').split(',')
            if value
        }
        allowed = {
            'answer_quality',
            'relevance',
            'clarity',
            'trust',
            'sources',
            'speed',
            'interaction',
            'no_preference',
            'other',
        }

        errors = {}
        if not selected:
            errors['pref03'] = 'Select at least one factor.'
        elif not selected.issubset(allowed):
            errors['pref03'] = 'One or more selected factors are invalid.'
        elif 'no_preference' in selected and len(selected) > 1:
            errors['pref03'] = (
                '"I had no preference" cannot be combined with other factors.'
            )

        if (
            'other' in selected
            and not str(values.get('pref03_other') or '').strip()
        ):
            errors['pref03_other'] = 'Please specify the other factor.'

        for field_name in ('pref03_other', 'pref04', 'fb01'):
            field_error = privacy_error(values.get(field_name))
            if field_error:
                errors[field_name] = field_error

        return errors or None

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.participant.completed_study = True


class Completion(Page):
    @staticmethod
    def is_displayed(player):
        return player.participant.vars.get('eligible_for_study', False)

    @staticmethod
    def vars_for_template(player):
        participant = player.participant
        return dict(
            progress_current=C.TOTAL_STUDY_PAGES,
            progress_total=C.TOTAL_STUDY_PAGES,
            progress_percent=100,
            study_title=participant.vars.get(
                'study_title',
                'AI-Assisted Bible Study: Comparing Two Chat Systems',
            ),
            researcher_name=participant.vars.get(
                'researcher_name',
                '[Your full name]',
            ),
            researcher_email=participant.vars.get(
                'researcher_email',
                '[Your university email]',
            ),
            support_resource_label=participant.vars.get(
                'support_resource_label',
                '',
            ),
            support_resource_url=participant.vars.get(
                'support_resource_url',
                '',
            ),
        )


def custom_export_anonymous_final_comparison(players):
    """Export completed final responses without oTree identity fields."""
    yield [
        'response_code',
        'sequence_id',
        'pref01',
        'pref03',
        'pref03_other',
        'pref04',
        'fb01',
    ]

    for player in players:
        participant = player.participant
        if not participant.completed_study:
            continue
        yield [
            participant.response_code,
            participant.sequence_id,
            player.field_maybe_none('pref01'),
            player.field_maybe_none('pref03'),
            player.field_maybe_none('pref03_other'),
            player.field_maybe_none('pref04'),
            player.field_maybe_none('fb01'),
        ]


page_sequence = [FinalComparison, Completion]
