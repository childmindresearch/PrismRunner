from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class FamilyHistoryCondition(str, Enum):
    GENERALIZED_ANXIETY_DISORDER = "generalized anxiety disorder"
    DEPRESSION = "depression"
    SUBSTANCE_ABUSE = "substance abuse"
    POSTTRAUMATIC_STRESS_DISORDER = "posttraumatic stress disorder"
    PSYCHOSIS = "psychosis"
    SUICIDALITY = "suicidality"
    EATING_DISORDER = "eating disorder"
    OBSESSIVE_COMPULSIVE_DISORDER = "obsessive compulsive disorder"
    BIPOLAR_DISORDER = "bipolar disorder"
    SCHIZOPHRENIA = "schizophrenia"
    OTHER = "other"


class PsychometricPerformanceLevel(str, Enum):
    EXTREMELY_LOW = "Extremely Low"
    LOW_AVERAGE = "Low Average"
    AVERAGE = "Average"
    HIGH_AVERAGE = "High Average"
    VERY_HIGH = "Very High"
    EXTREMELY_HIGH = "Extremely High"
    OTHER = "Other"


class ExecutiveFunctionSymptom(str, Enum):
    IMPAIRED_WORKING_MEMORY = "Impaired Working Memory"
    DEFICIT_IN_ATTENTION = "Deficit in Attention"
    REDUCED_MENTAL_FLEXIBILITY = "Reduced Mental Flexibility"
    SLOW_PROCESSING_SPEED = "Slow Processing Speed"
    OTHER = "Other"


class PsychometricIndex(str, Enum):
    VERBAL_COMPREHENSION_INDEX = "Verbal Comprehension Index"
    VISUAL_SPATIAL_INDEX = "Visual Spatial Index"
    FLUID_REASONING_INDEX = "Fluid Reasoning Index"
    WORKING_MEMORY_INDEX = "Working Memory Index"
    PROCESSING_SPEED_INDEX = "Processing Speed Index"
    GENERAL_ABILITY_INDEX = "General Ability Index"
    FULL_SCALE_IQ = "Full Scale IQ"
    PERFORMANCE_IQ = "Performance IQ"
    VERBAL_IQ = "Verbal IQ"
    OTHER = "Other"


class PhonologicalProcessingDomain(str, Enum):
    PHONOLOGICAL_AWARENESS = "phonological awareness"
    PHONOLOGICAL_MEMORY = "phonological memory"
    ORTHOGRAPHIC_PROCESSING = "orthographic processing"
    OTHER = "other"


class SpeechAssessmentResult(str, Enum):
    NORMAL_SPEECH = "normal speech"
    ARTICULATION_DIFFICULTY = "articulation difficulty"
    LISPS = "lisps"
    OTHER = "other"


class TechnologyType(str, Enum):
    SOFTWARE = "software"
    VIDEOGAME = "videogame"
    APP = "app"
    WEBSITE = "website"
    OTHER = "other"


class FeatureCategory(str, Enum):
    READING_SUPPORT = "reading_support"
    WRITING_ASSISTANCE = "writing_assistance"
    TASK_MANAGEMENT = "task_management"
    TEXT_TO_SPEECH = "text_to_speech"
    HIGHLIGHTING = "highlighting"
    ORGANIZERS = "organizers"
    OTHER = "other"


class PlatformType(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    WEB = "web"
    OTHER = "other"


class MedicationType(str, Enum):
    STIMULANT = "stimulant"
    ALPHA_AGONIST = "alpha_agonist"
    ATOMOXETINE = "atomoxetine"
    SSRI = "ssri"
    OTHER = "other"


class SideEffect(str, Enum):
    DECREASED_APPETITE = "decreased appetite"
    INSOMNIA = "insomnia"
    LOW_BLOOD_PRESSURE = "low blood pressure"
    INCREASED_HEART_RATE = "increased heart rate"
    MILD_SLEEPINESS = "mild sleepiness"
    OTHER = "other"


class MedicationHolidayStatus(str, Enum):
    RECOMMENDED = "recommended"
    NOT_RECOMMENDED = "not recommended"
    OPTIONAL = "optional"
    OTHER = "other"


class InterventionType(str, Enum):
    COGNITIVE_BEHAVIORAL_THERAPY = "Cognitive Behavioral Therapy"
    ACCEPTANCE_AND_COMMITMENT_THERAPY = "Acceptance and Commitment Therapy"
    PARENT_BASED_TREATMENT = "Parent-Based Treatment"
    PSYCHIATRIC_PHARMACOTHERAPY = "Psychiatric Pharmacotherapy"
    EDUCATIONAL_ACCOMMODATIONS = "Educational Accommodations"
    EXECUTIVE_FUNCTION_SUPPORT = "Executive Function Support"
    OTHER = "Other"


class AnxietyDisorder(str, Enum):
    GENERALIZED_ANXIETY_DISORDER = "Generalized Anxiety Disorder"
    SEPARATION_ANXIETY_DISORDER = "Separation Anxiety Disorder"
    SOCIAL_ANXIETY_DISORDER = "Social Anxiety Disorder"
    PANIC_DISORDER = "Panic Disorder"
    SPECIFIC_phobia = "Specific Phobia"
    OTHER = "Other"


class AssessmentTool(str, Enum):
    SCARED = "SCARED"
    DISS = "DISS"
    KIDDIE_SADS = "KIDDIE-SADS"
    OTHER = "Other"


class AccommodationType(str, Enum):
    EXTENDED_TIME = "extended_time"
    WORKSHEET_ACCOMMODATION = "worksheet_accommodation"
    BREAK_SCHEDULE = "break_schedule"
    TASK_BREAKDOWN = "task_breakdown"
    STRUCTURED_CLASSROOM = "structured_classroom"
    ASSISTIVE_TECHNOLOGY = "assistive_technology"
    OTHER = "other"


class ExecutiveFunctionSupport(str, Enum):
    EXTERNAL_PROMPTING = "external_prompting"
    FREQUENT_BREAKS = "frequent_breaks"
    ENVIRONMENTAL_STRUCTURE = "environmental_structure"
    BEHAVIORAL_REGULATION = "behavioral_regulation"
    OTHER = "other"


class InstructionalApproach(str, Enum):
    DIRECT_EXPLICIT = "direct_explicit"
    MULTISENSORY = "multisensory"
    VISUAL = "visual"
    KINESTHETIC = "kinesthetic"
    AUDITORY = "auditory"
    OTHER = "other"


class StrategyType(str, Enum):
    METACOGNITIVE = "metacognitive"
    BEHAVIORAL = "behavioral"
    EXECUTIVE_FUNCTION = "executive_function"
    OTHER = "other"


class LearningSupportStrategyType(str, Enum):
    IMPULSE_CONTROL = "impulse_control"
    ARITHMETIC_COMPUTATION = "arithmetic_computation"
    VISUAL_LEARNING_AID = "visual_learning_aid"
    INSTRUCTIONAL_AID = "instructional_aid"
    OTHER = "other"


class CopingStrategyType(Enum):
    RELAXATION_TECHNIQUE = "Relaxation Technique"
    PHYSICAL_EXERCISE = "Physical Exercise"
    SPIRITUAL_MANTRA = "Spiritual Mantra"
    JOURNALING = "Journaling"
    ROUTINE_MAINTENANCE = "Routine Maintenance"
    OTHER = "Other"


class TestEnvironmentSetup(str, Enum):
    QUIET_ENVIRONMENT = "quiet environment"
    MINIMAL_DISTRACTIONS = "minimal distractions"
    COMFORTABLE_SETTING = "comfortable setting"
    OTHER = "other"


class TestAdministrationProcedure(str, Enum):
    STANDARDIZED_PROCEDURES = "standardized procedures"
    MODIFIED_PROCEDURES = "modified procedures"
    OTHER = "other"


class ExaminerTrainingLevel(str, Enum):
    BASIC_TRAINING = "basic training"
    ADVANCED_TRAINING = "advanced training"
    SPECIALIZED_TRAINING = "specialized training"
    OTHER = "other"


class TestMaterialProtection(str, Enum):
    PLASTIC_SHEET_PROTECTORS = "plastic sheet protectors"
    HAND_SANITIZER_PROTOCOL = "hand sanitizer protocol"
    GLOVE_USAGE = "glove usage"
    OTHER = "other"


class SessionDuration(str, Enum):
    TWO_SESSIONS = "two sessions"
    THREE_SESSIONS = "three sessions"
    OTHER = "other"


class MedicalHistoryStatus(str, Enum):
    UNREMARKABLE = "unremarkable"
    REMARKABLE = "remarkable"
    OTHER = "other"


class MoodState(str, Enum):
    EUTHYMIC = "euthymic"
    OTHER = "other"


class ThoughtProcess(str, Enum):
    LOGICAL = "logical"
    FLIGHT_OF_IDEAS = "flight_of_ideas"
    LOOSE_ASSOCIATIONS = "loose_associations"
    OTHER = "other"


class DietaryHabit(str, Enum):
    HEALTHY_CHOICES = "healthy_choices"
    LIMITED_FRIED_FOODS = "limited_fried_foods"
    LIMITED_ADDED_SUGARS = "limited_added_sugars"
    LIMITED_SUGARY_DRINKS = "limited_sugary_drinks"
    OTHER = "other"


class CaregiverSupportStatus(str, Enum):
    ENCOURAGED_CONTACT = "encouraged_contact"
    NO_ACTION_NEEDED = "no_action_needed"
    OTHER = "other"


class SubscriptionPlan(str, Enum):
    ALLEY_MEMBERSHIP = "alley_membership"
    UNLIMITED_BOOKS = "unlimited_books"
    OTHER = "other"


class SupportResourceType(str, Enum):
    TIME_TIMER = "Time Timer"
    BEHAVIOR_CHART = "Behavior Chart"
    WORKSHOP = "Workshop"
    OTHER = "Other"


class DevelopmentalHistory(BaseModel):
    """Represents longitudinal biopsychosocial developmental history including family background, onset of symptoms, and environmental factors."""

    family_history_conditions: Optional[List[FamilyHistoryCondition]] = Field(
        default=None,
        description="List of psychiatric conditions in first, second, and third-degree relatives.",
    )
    symptom_onset_age: Optional[int] = Field(
        default=None, description="Age when symptoms were first noted."
    )
    symptom_duration: Optional[str] = Field(
        default=None,
        description="Duration of symptoms (e.g., since childhood, since age X).",
    )
    symptom_impairment_settings: Optional[List[str]] = Field(
        default=None,
        description="Settings where symptoms cause functional impairment (e.g., home, school, family).",
    )
    family_structure: Optional[str] = Field(
        default=None,
        description="Description of family structure (e.g., intact, bilingual household).",
    )
    family_functioning: Optional[str] = Field(
        default=None, description="Overall family functioning description."
    )
    environmental_stressors: Optional[str] = Field(
        default=None,
        description="Environmental stressors experienced during development (e.g., pandemic, trauma).",
    )
    medical_conditions: Optional[List[str]] = Field(
        default=None,
        description="Medical conditions relevant to developmental history (e.g., KBG syndrome).",
    )
    language_background: Optional[List[str]] = Field(
        default=None, description="Languages spoken in the household."
    )
    referral_source: Optional[str] = Field(
        default=None,
        description="How the family became involved in the study or care process.",
    )
    caregiver_concerns: Optional[str] = Field(
        default=None,
        description="Primary concerns expressed by caregivers regarding the individual's development.",
    )
    behavioral_patterns: Optional[List[str]] = Field(
        default=None,
        description="Noted behavioral patterns or incidents (e.g., wetting accidents, outbursts).",
    )
    symptom_context: Optional[str] = Field(
        default=None,
        description="Contextual information about when symptoms typically occur.",
    )
    impact_on_daily_life: Optional[str] = Field(
        default=None, description="Impact of symptoms on daily life activities."
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any additional relevant notes about developmental history.",
    )


class PsychometricAssessment(BaseModel):
    """Represents a comprehensive psychometric assessment report focusing on cognitive abilities, executive functions, and academic performance."""

    patient_id: Optional[str] = Field(
        default=None, description="Unique identifier for the patient."
    )
    indexs: Optional[List[str]] = Field(default=None, description="Associated indexs.")
    iqs: Optional[List[str]] = Field(default=None, description="Associated iqs.")
    performance_level: Optional[PsychometricPerformanceLevel] = Field(
        default=None,
        description="Overall performance level according to standardized ranges.",
    )
    executive_function_symptoms: Optional[List[ExecutiveFunctionSymptom]] = Field(
        default=None, description="List of identified executive dysfunction symptoms."
    )
    academic_achievement_subtests: Optional[List[str]] = Field(
        default=None, description="Subtests used to evaluate academic progress."
    )
    assessment_tools_used: Optional[List[str]] = Field(
        default=None,
        description="Tools and instruments utilized during the assessment.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional comments or observations from the assessment.",
    )


class EducationalSupportProgram(BaseModel):
    """Represents an educational support program designed to assist students with learning challenges."""

    name: Optional[str] = Field(
        default=None, description="Name of the educational support program."
    )
    description: Optional[str] = Field(
        default=None, description="Description of the program's purpose and components."
    )
    target_skills: Optional[List[str]] = Field(
        default=None,
        description="Skills targeted by the program such as phonological awareness or spelling.",
    )
    accommodations: Optional[List[str]] = Field(
        default=None, description="Specific accommodations provided by the program."
    )
    assistive_technologies: Optional[List[str]] = Field(
        default=None,
        description="Technologies used in the program such as text-to-speech apps.",
    )
    intervention_approach: Optional[str] = Field(
        default=None,
        description="General approach taken by the program (e.g., multi-modal, structured).",
    )
    focus_areas: Optional[List[str]] = Field(
        default=None, description="Areas of focus like reading, writing, or vocabulary."
    )
    recommended_for: Optional[List[str]] = Field(
        default=None,
        description="Student profiles or conditions the program is recommended for.",
    )
    duration: Optional[str] = Field(
        default=None, description="Duration or time frame of the program."
    )
    materials_needed: Optional[List[str]] = Field(
        default=None, description="Materials required to implement the program."
    )
    assessment_tools: Optional[List[str]] = Field(
        default=None, description="Tools used for assessing student progress."
    )
    progress_checkpoints: Optional[List[str]] = Field(
        default=None, description="Methods or intervals for checking student progress."
    )
    customization_options: Optional[List[str]] = Field(
        default=None, description="Options available for customizing the program."
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any additional notes regarding implementation or effectiveness.",
    )


class ClinicalAssessment(BaseModel):
    """Captures clinical psychometric assessments related to phonological processing and speech sound disorders."""

    assessment_type: Optional[str] = Field(
        default=None, description="Type of assessment performed (e.g., CTOPP-2, RAS)."
    )
    domains_assessed: Optional[List[PhonologicalProcessingDomain]] = Field(
        default=None,
        description="Domains assessed in the phonological processing evaluation.",
    )
    results_summary: Optional[str] = Field(
        default=None, description="Summary of key findings from the assessment."
    )
    speech_sound_disorder: Optional[SpeechAssessmentResult] = Field(
        default=None,
        description="Result indicating presence or absence of speech sound disorder.",
    )
    phonological_processing_score: Optional[str] = Field(
        default=None,
        description="Score or level of phonological processing proficiency.",
    )
    reading_related_findings: Optional[str] = Field(
        default=None,
        description="Findings specifically related to reading performance.",
    )
    recommendations: Optional[List[str]] = Field(
        default=None, description="Recommendations based on the assessment results."
    )
    next_steps: Optional[List[str]] = Field(
        default=None, description="Next steps suggested after the assessment."
    )
    assessment_date: Optional[str] = Field(
        default=None, description="Date when the assessment was conducted."
    )
    evaluator: Optional[str] = Field(
        default=None,
        description="Evaluator or organization responsible for the assessment.",
    )
    supporting_evidence: Optional[List[str]] = Field(
        default=None, description="Evidence supporting the assessment conclusions."
    )
    special_considerations: Optional[str] = Field(
        default=None, description="Special considerations noted during the assessment."
    )
    follow_up_required: Optional[bool] = Field(
        default=None, description="Indicates if a follow-up assessment is needed."
    )


class AssistiveTechnology(BaseModel):
    """Represents an assistive technology product or service designed to support educational accommodations and cognitive scaffolding."""

    name: Optional[str] = Field(
        default=None, description="Name of the assistive technology"
    )
    description: Optional[str] = Field(
        default=None, description="Description of the technology's purpose and features"
    )
    technology_type: Optional[TechnologyType] = Field(
        default=None, description="Type of technology (e.g., software, videogame)"
    )
    platform_types: Optional[List[PlatformType]] = Field(
        default=None, description="Supported platforms (e.g., mobile, web)"
    )
    price: Optional[str] = Field(
        default=None, description="Pricing information if applicable"
    )
    features: Optional[List[FeatureCategory]] = Field(
        default=None, description="Key functional categories supported"
    )
    url: Optional[str] = Field(
        default=None, description="URL to the official website or app store"
    )
    membership_required: Optional[bool] = Field(
        default=None,
        description="Whether a membership is required to access the service",
    )
    free_version_limit: Optional[str] = Field(
        default=None,
        description="Limitations of the free version, e.g., number of documents",
    )
    accessibility_features: Optional[List[str]] = Field(
        default=None, description="Specific accessibility features offered"
    )
    target_audience: Optional[str] = Field(
        default=None, description="Target demographic or user group"
    )
    review_summary: Optional[str] = Field(
        default=None, description="Summary of professional reviews or ratings"
    )
    additional_resources: Optional[List[str]] = Field(
        default=None,
        description="Links to additional resources like articles or guides",
    )
    supported_languages: Optional[List[str]] = Field(
        default=None, description="Languages supported by the technology"
    )
    developer_organization: Optional[str] = Field(
        default=None,
        description="Organization responsible for developing the technology",
    )


class ADHDPharmacotherapy(BaseModel):
    """Represents the clinical management protocol for ADHD pharmacotherapy including medication types, dosing considerations, monitoring requirements, and side effect profiles."""

    medication_type: Optional[MedicationType] = Field(
        default=None, description="Type of ADHD medication prescribed"
    )
    dosage_mg: Optional[float] = Field(
        default=None, description="Dosage of the medication in milligrams"
    )
    side_effects: Optional[List[SideEffect]] = Field(
        default=None, description="List of reported side effects"
    )
    monitoring_requirements: Optional[List[str]] = Field(
        default=None, description="Vital signs or health metrics to monitor"
    )
    onset_time_hours: Optional[float] = Field(
        default=None, description="Time in hours for medication to take effect"
    )
    duration_weeks: Optional[float] = Field(
        default=None, description="Duration in weeks for full therapeutic effect"
    )
    medication_holiday_status: Optional[MedicationHolidayStatus] = Field(
        default=None, description="Recommendation status for medication holidays"
    )
    effectiveness_rating: Optional[str] = Field(
        default=None, description="Rating of medication effectiveness"
    )
    discontinuation_reason: Optional[str] = Field(
        default=None, description="Reason for discontinuing the medication"
    )
    alternative_treatments: Optional[List[str]] = Field(
        default=None, description="Alternative or adjunct treatments considered"
    )
    patient_response_notes: Optional[str] = Field(
        default=None, description="Notes on patient's response to treatment"
    )
    contraindications: Optional[List[str]] = Field(
        default=None, description="Known contraindications or drug interactions"
    )
    special_considerations: Optional[List[str]] = Field(
        default=None,
        description="Special considerations for administration or monitoring",
    )
    follow_up_timeline_days: Optional[int] = Field(
        default=None, description="Days until next follow-up appointment"
    )
    additional_comments: Optional[str] = Field(
        default=None, description="Additional clinical comments or notes"
    )


class PediatricAnxietyIntervention(BaseModel):
    """A structured model representing the components of pediatric anxiety intervention including therapies, assessments, and support systems."""

    intervention_type: Optional[InterventionType] = Field(
        default=None, description="The type of intervention being applied."
    )
    anxiety_disorders: Optional[List[AnxietyDisorder]] = Field(
        default=None, description="List of diagnosed anxiety disorders."
    )
    assessment_tools_used: Optional[List[AssessmentTool]] = Field(
        default=None, description="Tools used for clinical assessment."
    )
    caregiver_support_provided: Optional[bool] = Field(
        default=None, description="Indicates if caregiver support was offered."
    )
    educational_accommodations: Optional[bool] = Field(
        default=None, description="Whether educational accommodations were implemented."
    )
    executive_function_scaffolding: Optional[bool] = Field(
        default=None, description="If scaffolding for executive functions was provided."
    )
    pharmacotherapy_considered: Optional[bool] = Field(
        default=None, description="Whether medication decisions were discussed."
    )
    parent_treatment_program: Optional[str] = Field(
        default=None,
        description="Name or reference to parent-based treatment programs.",
    )
    recommended_strategies: Optional[List[str]] = Field(
        default=None, description="Strategies recommended for managing anxiety."
    )
    key_resources: Optional[Dict[str, str]] = Field(
        default=None, description="Key resources such as books or websites."
    )
    clinical_interview_summary: Optional[str] = Field(
        default=None, description="Summary of clinical interview findings."
    )
    diagnostic_classification: Optional[str] = Field(
        default=None,
        description="Diagnostic classification from structured interviews.",
    )
    treatment_outcome_notes: Optional[str] = Field(
        default=None, description="Notes on treatment outcomes or progress."
    )
    additional_comments: Optional[str] = Field(
        default=None, description="Any other relevant comments or observations."
    )


class EducationalAccommodationPlan(BaseModel):
    """A comprehensive plan outlining educational accommodations and support strategies tailored for students with executive functioning challenges."""

    student_profile: Optional[str] = Field(
        default=None,
        description="Description of the student's learning profile and identified difficulties.",
    )
    accommodation_types: Optional[List[AccommodationType]] = Field(
        default=None,
        description="Types of accommodations implemented to support the student.",
    )
    executive_function_supports: Optional[List[ExecutiveFunctionSupport]] = Field(
        default=None,
        description="Strategies aimed at scaffolding executive function skills.",
    )
    assistive_technology_tools: Optional[List[str]] = Field(
        default=None,
        description="Specific assistive technology tools used, e.g., SnapType Pro.",
    )
    break_schedule: Optional[str] = Field(
        default=None,
        description="Details regarding scheduled breaks during tasks or tests.",
    )
    task_management_strategies: Optional[str] = Field(
        default=None,
        description="Methods for breaking down large tasks into smaller, manageable units.",
    )
    environmental_structures: Optional[str] = Field(
        default=None,
        description="Structural elements in the classroom or learning environment that aid focus.",
    )
    external_support_mechanisms: Optional[str] = Field(
        default=None,
        description="Mechanisms such as prompts or peer modeling to initiate tasks.",
    )
    behavioral_regulation_approaches: Optional[str] = Field(
        default=None,
        description="Approaches for managing behavior and emotional regulation.",
    )
    caregiver_advocacy_resources: Optional[str] = Field(
        default=None,
        description="Resources or programs designed to support caregivers and advocates.",
    )
    assessment_findings: Optional[str] = Field(
        default=None,
        description="Summary of clinical psychometric assessments related to attention and executive functioning.",
    )
    support_plan_context: Optional[str] = Field(
        default=None,
        description="Contextual information about where and how the support plan is applied.",
    )
    recommended_interventions: Optional[List[str]] = Field(
        default=None,
        description="Recommended interventions based on observed symptoms and needs.",
    )
    strengths_recognition: Optional[str] = Field(
        default=None,
        description="How student strengths are acknowledged and reinforced in the curriculum.",
    )
    implementation_notes: Optional[str] = Field(
        default=None,
        description="Additional notes on implementation or modifications needed.",
    )


class EducationalSupportPlan(BaseModel):
    """A comprehensive plan outlining educational accommodations, instructional strategies, and cognitive scaffolding techniques tailored for students requiring additional support."""

    instructional_approaches: Optional[List[InstructionalApproach]] = Field(
        default=None,
        description="List of instructional methodologies employed such as direct explicit, multisensory, etc.",
    )
    strategy_types: Optional[List[StrategyType]] = Field(
        default=None,
        description="Types of strategies implemented including metacognitive, behavioral, executive function.",
    )
    modified_assignments: Optional[List[str]] = Field(
        default=None,
        description="Examples of adjusted tasks like reduced length reports or fewer test items.",
    )
    visual_cues: Optional[List[str]] = Field(
        default=None,
        description="Visual indicators or aids used to draw attention or convey information.",
    )
    auditory_methods: Optional[List[str]] = Field(
        default=None,
        description="Methods involving sound or speech, e.g., reading aloud, taped books.",
    )
    kinesthetic_activities: Optional[List[str]] = Field(
        default=None,
        description="Physical movement-based learning activities such as writing motor patterns.",
    )
    goal_setting: Optional[str] = Field(
        default=None,
        description="Description of how goals are framed, e.g., accuracy vs. speed.",
    )
    error_correction_techniques: Optional[List[str]] = Field(
        default=None,
        description="Techniques aimed at increasing error recognition and correction.",
    )
    prediction_practices: Optional[List[str]] = Field(
        default=None,
        description="Activities where students predict performance outcomes and compare with results.",
    )
    behavioral_instruction: Optional[str] = Field(
        default=None,
        description="Guidelines for delivering clear directives to students.",
    )
    material_checklists: Optional[bool] = Field(
        default=None,
        description="Whether daily material checklists are provided to students.",
    )
    oral_directions_limit: Optional[int] = Field(
        default=None,
        description="Maximum number of steps recommended in oral instructions.",
    )
    assessment_category: Optional[str] = Field(
        default=None,
        description="Category of clinical psychometric assessment if applicable.",
    )


class LearningSupportStrategy(BaseModel):
    """A model representing educational accommodations, assistive technologies, and cognitive support strategies for learners."""

    strategy_type: LearningSupportStrategyType = Field(
        ..., description="The category of the learning support strategy"
    )
    description: Optional[str] = Field(
        None, description="Detailed description of the strategy"
    )
    key_words: Optional[List[str]] = Field(
        None, description="Important keywords associated with the strategy"
    )
    visual_aid: Optional[str] = Field(
        None, description="Description of visual aids used"
    )
    step_sequence: Optional[List[str]] = Field(
        None, description="Sequence of steps for procedural strategies"
    )
    memory_technique: Optional[str] = Field(
        None, description="Memory strategy for algorithm recall"
    )
    color_coding: Optional[str] = Field(
        None, description="Color coding instructions if applicable"
    )
    socialization_opportunity: Optional[str] = Field(
        None, description="Details about social interaction support"
    )
    emotional_regulation_tactic: Optional[str] = Field(
        None, description="Techniques for managing emotions and impulses"
    )
    peer_collaboration_activity: Optional[str] = Field(
        None, description="Peer-based collaborative learning activity"
    )
    conceptual_reference: Optional[str] = Field(
        None, description="Reference material or concept being supported"
    )
    repetition_encouragement: Optional[str] = Field(
        None, description="Encouragement for repeating successful behaviors"
    )
    consequence_evaluation: Optional[str] = Field(
        None, description="Guidance on evaluating outcomes"
    )
    alternative_representation: Optional[str] = Field(
        None, description="Alternative ways to represent information"
    )
    example_application: Optional[str] = Field(
        None, description="Example usage of the strategy"
    )


class BehavioralRegulationPlan(BaseModel):
    """A comprehensive plan for managing emotional regulation and behavioral responses through structured interventions."""

    person_name: Optional[str] = Field(
        default=None, description="Name of the individual involved in the plan."
    )
    coping_strategies: Optional[List[CopingStrategyType]] = Field(
        default=None,
        description="List of coping strategies tailored to the individual's needs.",
    )
    relaxation_techniques: Optional[List[str]] = Field(
        default=None,
        description="Specific relaxation techniques recommended for stress reduction.",
    )
    physical_exercises: Optional[List[str]] = Field(
        default=None,
        description="Recommended physical exercises to alleviate anxiety and promote calmness.",
    )
    spiritual_mantras: Optional[List[str]] = Field(
        default=None,
        description="Mantras or affirmations intended to focus the mind and reduce anxiety.",
    )
    journaling_prompts: Optional[List[str]] = Field(
        default=None,
        description="Guided prompts for journaling to process thoughts and emotions.",
    )
    safe_spaces: Optional[List[str]] = Field(
        default=None,
        description="Locations where the individual can retreat to calm down or escape overwhelming situations.",
    )
    routine_elements: Optional[List[str]] = Field(
        default=None,
        description="Essential elements of daily routines that foster a sense of security.",
    )
    assessment_notes: Optional[str] = Field(
        default=None,
        description="Notes regarding clinical assessments or diagnostic classifications relevant to the plan.",
    )
    intervention_goals: Optional[List[str]] = Field(
        default=None,
        description="Primary objectives of the behavioral intervention approach.",
    )
    additional_supports: Optional[List[str]] = Field(
        default=None,
        description="Supplementary supports such as assistive technologies or accommodations.",
    )
    stress_reduction_methods: Optional[List[str]] = Field(
        default=None, description="Methods aimed at reducing overall stress levels."
    )
    muscle_tension_relief: Optional[bool] = Field(
        default=None,
        description="Indicates whether progressive muscle relaxation is part of the regimen.",
    )
    self_soothing_skills: Optional[bool] = Field(
        default=None,
        description="Whether self-soothing skills training is included in the plan.",
    )
    reward_consequence_system: Optional[bool] = Field(
        default=None,
        description="Indicates implementation of reward and consequence systems for behavior management.",
    )


class SleepWakeTransitionSupport(BaseModel):
    """A comprehensive model for supporting healthy sleep-wake transitions through behavioral, environmental, and therapeutic interventions."""

    person_name: Optional[str] = Field(
        default=None, description="Name of the individual receiving support"
    )
    consistency_practice: Optional[str] = Field(
        default=None, description="Reinforces body's sleep-wake cycle"
    )
    natural_light_exposure: Optional[bool] = Field(
        default=None, description="Exposure to natural light and fresh air"
    )
    routine_activities: Optional[List[str]] = Field(
        default=None, description="Activities done nightly to signal bedtime"
    )
    leisure_activity_buffer: Optional[bool] = Field(
        default=None, description="Minutes of downtime between activities"
    )
    relaxation_activities: Optional[List[str]] = Field(
        default=None, description="Activities promoting sleep transition"
    )
    electronic_device_removal: Optional[bool] = Field(
        default=None, description="Removal of electronic devices during relaxation"
    )
    sensory_interventions: Optional[List[str]] = Field(
        default=None, description="Interventions affecting sensory input"
    )
    environmental_conditions: Optional[Dict[str, str]] = Field(
        default=None, description="Room conditions for optimal sleep"
    )
    medication_consideration: Optional[str] = Field(
        default=None, description="Potential pharmacotherapy options if needed"
    )
    caregiver_support: Optional[bool] = Field(
        default=None, description="Support provided to caregivers"
    )
    educational_accommodations: Optional[List[str]] = Field(
        default=None, description="Accommodations for academic settings"
    )
    assistive_devices: Optional[List[str]] = Field(
        default=None, description="Devices aiding independence and performance"
    )
    extended_time_accommodation: Optional[bool] = Field(
        default=None, description="Extended time for assessments or tasks"
    )
    other_support_strategies: Optional[List[str]] = Field(
        default=None, description="Additional strategies not covered above"
    )


class ClinicalAssessmentReport(BaseModel):
    """A comprehensive report detailing clinical psychometric assessment procedures, environmental setup, examiner qualifications, and developmental history for diagnostic classification."""

    test_environment_setup: Optional[List[TestEnvironmentSetup]] = Field(
        default=None,
        description="List of environmental setup elements such as spacing of furniture and noise control.",
    )
    test_administration_procedure: Optional[TestAdministrationProcedure] = Field(
        default=None, description="Type of procedure used during test administration."
    )
    examiner_training_level: Optional[ExaminerTrainingLevel] = Field(
        default=None,
        description="Level of examiner training applied during assessment.",
    )
    test_material_protection: Optional[List[TestMaterialProtection]] = Field(
        default=None,
        description="Measures taken to sanitize and protect test materials.",
    )
    session_duration: Optional[SessionDuration] = Field(
        default=None,
        description="Number of sessions required to complete the assessment.",
    )
    medical_history_status: Optional[MedicalHistoryStatus] = Field(
        default=None,
        description="Overall status of the patient's medical history relevant to assessment.",
    )
    accommodations_and_technologies: Optional[List[AccommodationType]] = Field(
        default=None,
        description="Types of educational accommodations or assistive technologies recommended.",
    )
    executive_function_support: Optional[List[ExecutiveFunctionSupport]] = Field(
        default=None,
        description="Cognitive supports provided to aid in executive function performance.",
    )
    diagnosis_criteria_met: Optional[bool] = Field(
        default=None,
        description="Indicates whether diagnostic criteria for ADHD or similar disorders were met.",
    )
    attention_monitoring_required: Optional[bool] = Field(
        default=None,
        description="Indicates if ongoing monitoring of attention is needed.",
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any other relevant information or observations from the assessment process.",
    )
    assessment_sessions_completed: Optional[int] = Field(
        default=None,
        description="Total number of sessions actually completed for the assessment.",
    )
    test_completion_context: Optional[str] = Field(
        default=None,
        description="Contextual details about where and how the tests were administered.",
    )
    key_findings_summary: Optional[str] = Field(
        default=None,
        description="Summary of key findings regarding cognitive abilities and academic skills.",
    )


class AssessmentSummary(BaseModel):
    """A comprehensive psychosocial assessment summary capturing clinical findings, dietary behaviors, caregiver support status, and educational accommodations."""

    mood_state: Optional[MoodState] = Field(
        default=None, description="The patient's current mood state."
    )
    thought_process: Optional[ThoughtProcess] = Field(
        default=None, description="Description of the patient's thought process."
    )
    dietary_habits: Optional[List[DietaryHabit]] = Field(
        default=None, description="List of dietary habits observed or recommended."
    )
    caregiver_support_status: Optional[CaregiverSupportStatus] = Field(
        default=None, description="Indicates if caregiver contact is encouraged."
    )
    subscription_plan: Optional[SubscriptionPlan] = Field(
        default=None,
        description="Type of subscription plan associated with assistive tools.",
    )
    contact_information: Optional[str] = Field(
        default=None, description="Contact details for the care provider."
    )
    ally_program_access: Optional[bool] = Field(
        default=None, description="Whether the user has access to the ally program."
    )
    educational_accommodations: Optional[str] = Field(
        default=None,
        description="Details about educational accommodations or assistive technologies used.",
    )
    biopsychosocial_history: Optional[str] = Field(
        default=None, description="Summary of longitudinal developmental history."
    )
    executive_function_support: Optional[str] = Field(
        default=None,
        description="Notes on scaffolding or cognitive support strategies.",
    )
    behavioral_regulation_therapy: Optional[str] = Field(
        default=None,
        description="Information on therapies related to emotional regulation.",
    )
    nutrition_importance: Optional[str] = Field(
        default=None,
        description="Statement on the importance of nutrition in treatment.",
    )
    diagnostic_classification: Optional[str] = Field(
        default=None,
        description="Current diagnostic classification or presumptive diagnosis.",
    )
    target_outcomes: Optional[List[str]] = Field(
        default=None, description="List of targeted outcomes based on assessment."
    )
    additional_notes: Optional[str] = Field(
        default=None, description="Any other relevant notes from the assessment."
    )


class CaregiverSupportPlan(BaseModel):
    """A comprehensive plan outlining psychoeducational caregiver support and advocacy strategies, executive function scaffolding techniques, and behavioral regulation therapies."""

    support_resource_type: Optional[SupportResourceType] = Field(
        default=None,
        description="The type of clinical support resource being utilized.",
    )
    resource_description: Optional[str] = Field(
        default=None,
        description="A brief description of the clinical support resource.",
    )
    temporal_measurement_method: Optional[str] = Field(
        default=None,
        description="Method used to measure or track time, such as visual timers or stopwatches.",
    )
    executive_function_strategy: Optional[str] = Field(
        default=None,
        description="Strategy employed to scaffold executive functions, e.g., using clocks or timers.",
    )
    behavioral_regulation_technique: Optional[str] = Field(
        default=None,
        description="Technique used for emotional and behavioral regulation, such as reinforcing effort.",
    )
    family_support_involvement: Optional[str] = Field(
        default=None,
        description="Description of how family support is integrated into the care plan.",
    )
    homework_monitoring_approach: Optional[str] = Field(
        default=None,
        description="Approach taken to monitor the quantity and completion of homework assignments.",
    )
    positive_clinical_outcome: Optional[str] = Field(
        default=None,
        description="Expected or observed positive clinical outcome from the intervention.",
    )
    educational_accommodation: Optional[str] = Field(
        default=None,
        description="Specific educational accommodation implemented for the individual.",
    )
    assistive_technology_used: Optional[str] = Field(
        default=None,
        description="Assistive technology tool used to support learning or behavior.",
    )
    caregiver_training_component: Optional[str] = Field(
        default=None,
        description="Component of training provided to caregivers or educators.",
    )
    reinforcement_strategy: Optional[str] = Field(
        default=None,
        description="Strategy for reinforcing effort and accepting imperfections.",
    )
    communication_tool: Optional[str] = Field(
        default=None,
        description="Tool used for communication between school and home, such as notes or charts.",
    )
    target_population: Optional[List[str]] = Field(
        default=None,
        description="List of target population groups involved in the support plan.",
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any other relevant information about the support plan.",
    )
