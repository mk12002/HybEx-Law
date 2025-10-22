% ============================================================================
% LEGAL AID ELIGIBILITY RULES - CORRECTED & SIMPLIFIED VERSION
% Based on: Legal Services Authorities Act, 1987
% ============================================================================

% ✅ Bug #9 Fix: Add missing discontiguous declarations
:- discontiguous eligible_with_confidence/3.
:- discontiguous generate_detailed_reasoning/2.
:- discontiguous primary_eligibility_reason/2.
:- discontiguous check_and_print_eligibility/1.
:- discontiguous format_eligibility_reason/4.
:- discontiguous not_eligible_with_confidence/3.
:- discontiguous eligible_with_confidence_enhanced/3.

:- dynamic has_income/1.
:- dynamic no_income/1.
:- dynamic vulnerable_group/2.
:- dynamic no_vulnerable_group/1.
:- dynamic person/1.
:- dynamic social_category/2.
:- dynamic annual_income/2.
:- dynamic monthly_income/2.

% ============================================================================
% INCOME THRESHOLD FACTS (Required by negative rules)
% ============================================================================
% These thresholds define maximum annual income for each social category
% to be eligible for legal aid under Legal Services Authorities Act, 1987

income_threshold(sc, 800000).      % Scheduled Caste: ₹8,00,000
income_threshold(st, 800000).      % Scheduled Tribe: ₹8,00,000
income_threshold(obc, 600000).     % Other Backward Classes: ₹6,00,000
income_threshold(bpl, 0).          % Below Poverty Line: No income limit
income_threshold(ews, 800000).     % Economically Weaker Section: ₹8,00,000
income_threshold(general, 300000). % General category: ₹3,00,000 (NALSA 2024)

% ============================================================================
% CORE ELIGIBILITY RULES (Priority Order - FIRST MATCH WINS)
% ============================================================================

% Rule 1: NO INCOME → ALWAYS ELIGIBLE (Highest Priority)
% Confidence: 0.95
eligible_with_confidence(Person, eligible, 0.95) :-
    no_income(Person), !.

% Rule 2: VULNERABLE GROUPS → ALWAYS ELIGIBLE
% Includes: disabled, women, senior_citizen, widow, single_parent, transgender, low_income
% Confidence: 0.95
eligible_with_confidence(Person, eligible, 0.95) :-
    vulnerable_group(Person, _), !.

% Rule 3: SC/ST with income ≤ ₹8,00,000
% Confidence: 0.90
eligible_with_confidence(Person, eligible, 0.90) :-
    (social_category(Person, sc) ; social_category(Person, st)),
    annual_income(Person, Income),
    Income =< 800000, !.

% Rule 4: OBC with income ≤ ₹6,00,000
% Confidence: 0.88
eligible_with_confidence(Person, eligible, 0.88) :-
    social_category(Person, obc),
    annual_income(Person, Income),
    Income =< 600000, !.

% Rule 5: General category with income ≤ ₹3,00,000
% Confidence: 0.85 (Aligned with NALSA 2024 and config.py)
eligible_with_confidence(Person, eligible, 0.85) :-
    social_category(Person, general),
    annual_income(Person, Income),
    Income =< 300000, !.

% Rule 6: EWS with income ≤ ₹8,00,000
% Confidence: 0.88
eligible_with_confidence(Person, eligible, 0.88) :-
    social_category(Person, ews),
    annual_income(Person, Income),
    Income =< 800000, !.

% Rule 7: BPL → ALWAYS ELIGIBLE
% Confidence: 0.95
eligible_with_confidence(Person, eligible, 0.95) :-
    social_category(Person, bpl), !.

% Rule 8: DEFAULT - NOT ELIGIBLE
% This catches all cases that don't match above rules
% Confidence: 0.70
eligible_with_confidence(_, not_eligible, 0.70).

% ============================================================================
% QUERY INTERFACE FOR PYTHON
% ============================================================================

% Main query predicate - prefer explicit negative rules first, then positive rules
check_and_print_eligibility(Person) :-
    % Check negative rules first
    ( not_eligible_with_confidence(Person, Decision, Confidence) ->
        writeln(Decision),
        writeln(Confidence)
    ; % Then check positive rules
      ( eligible_with_confidence(Person, Decision, Confidence) ->
            writeln(Decision),
            writeln(Confidence)
      ; % Default: not eligible with neutral confidence
            writeln(not_eligible),
            writeln(0.50)
      )
    ), !.

% Generate detailed reasoning string
generate_detailed_reasoning(Person, Reason) :-
    eligible_with_confidence(Person, Decision, _),
    (Decision = eligible -> 
        Reason = 'Eligible under Legal Services Authorities Act, 1987'
    ;   Reason = 'Does not meet eligibility criteria'), !.

generate_detailed_reasoning(_, 'Unable to determine eligibility').

% Get primary reason for eligibility decision
primary_eligibility_reason(Person, Reason) :-
    (no_income(Person) ->
        Reason = 'No income - automatically eligible'
    ; vulnerable_group(Person, Group) ->
        atom_concat('Eligible as vulnerable group: ', Group, Reason)
    ; annual_income(Person, Income), social_category(Person, Category) ->
        format(atom(Reason), 'Income ₹~w for category ~w', [Income, Category])
    ;   Reason = 'Does not meet eligibility criteria'
    ), !.

primary_eligibility_reason(_, 'Unable to determine reason').
% Threshold: Annual income ≤ ₹6,00,000
% ----------------------------------------------------------------------------
eligible_with_confidence(Person, eligible, 0.88) :-
    social_category(Person, obc),
    annual_income(Person, Income),
    Income =< 600000,
    !.

% ----------------------------------------------------------------------------
% Rule 6: EWS (Economically Weaker Section)
% Priority: HIGH (0.88 confidence)
% Threshold: Annual income ≤ ₹8,00,000
% ----------------------------------------------------------------------------
eligible_with_confidence(Person, eligible, 0.88) :-
    social_category(Person, ews),
    annual_income(Person, Income),
    Income =< 800000,
    !.

% ----------------------------------------------------------------------------
% Rule 7: BPL (Below Poverty Line) → ALWAYS ELIGIBLE
% Priority: VERY HIGH (0.95 confidence)
% ----------------------------------------------------------------------------
eligible_with_confidence(Person, eligible, 0.95) :-
    social_category(Person, bpl),
    !.

% ----------------------------------------------------------------------------
% Rule 8: DEFAULT - NOT ELIGIBLE
% Priority: LOWEST (0.70 confidence)
% This is the catch-all rule if none of the above match
% ----------------------------------------------------------------------------
eligible_with_confidence(Person, not_eligible, 0.70).

% ============================================================================
% HELPER PREDICATES FOR INCOME THRESHOLD CHECKS
% ============================================================================

% Check if person meets income threshold for their category
meets_income_threshold(Person, Category, Threshold) :-
    annual_income(Person, Income),
    Income =< Threshold.

% Check if person has income below general threshold (₹3,00,000 - NALSA 2024)
below_general_threshold(Person) :-
    annual_income(Person, Income),
    Income =< 300000.

% ============================================================================
% VULNERABLE GROUP ELIGIBILITY HELPER
% ============================================================================

% A person is vulnerable group eligible if they belong to ANY vulnerable group
vulnerable_group_eligible(Person) :-
    vulnerable_group(Person, _),
    \+ no_vulnerable_group(Person).

% ============================================================================
% DETAILED REASONING GENERATION
% ============================================================================

generate_detailed_reasoning(Person, Reason) :-
    eligible_with_confidence(Person, Decision, Confidence),
    format_eligibility_reason(Person, Decision, Confidence, Reason).

% ✅ Fixed: All variables are used, added cut to prevent backtracking
format_eligibility_reason(Person, eligible, Confidence, Reason) :-
    Confidence >= 0.95,
    (   no_income(Person) ->
        Reason = 'Eligible: No income reported (Section 12 LSA Act)'
    ;   vulnerable_group(Person, Group) ->
        atomic_list_concat(['Eligible: Member of vulnerable group (', Group, ')'], Reason)
    ;   social_category(Person, bpl) ->
        Reason = 'Eligible: Below Poverty Line (BPL) category'
    ;   Reason = 'Eligible: High confidence based on multiple factors'
    ), !.

% Line 192 - Use the Confidence parameter, added cut
format_eligibility_reason(Person, eligible, Confidence, Reason) :-
    Confidence < 0.95,
    annual_income(Person, Income),
    social_category(Person, Category),
    format(atom(Reason), 'Eligible: ~w category with annual income ₹~w (confidence: ~2f)', 
           [Category, Income, Confidence]), !.

% Line 198 - Use the Confidence parameter, added cut
format_eligibility_reason(Person, not_eligible, Confidence, Reason) :-
    (   annual_income(Person, Income),
        social_category(Person, Category) ->
        format(atom(Reason), 'Not eligible: ~w category income ₹~w exceeds threshold (confidence: ~2f)', 
               [Category, Income, Confidence])
    ;   format(atom(Reason), 'Not eligible: Does not meet eligibility criteria (confidence: ~2f)', 
               [Confidence])
    ), !.

% ============================================================================
% PRIMARY ELIGIBILITY REASON (SHORT FORM)
% ============================================================================

primary_eligibility_reason(Person, Reason) :-
    eligible_with_confidence(Person, Decision, _),
    (   Decision = eligible ->
        Reason = 'Eligible under Legal Services Authorities Act, 1987'
    ;   Reason = 'Does not meet eligibility criteria'
    ).

% NOTE: Query interface defined earlier (check_and_print_eligibility/1) -
% negative-first behavior implemented above to prefer explicit non-eligibility checks.
% ============================================================================
% ENHANCED LEGAL AID ELIGIBILITY RULES WITH NEGATIVE RULES
% ============================================================================

% ============================================================================
% NEGATIVE ELIGIBILITY RULES (When NOT eligible - explicit)
% ============================================================================

% Rule N1: High income without vulnerable status = NOT ELIGIBLE
not_eligible_with_confidence(Person, not_eligible, 0.95) :-
    has_income(Person),
    annual_income(Person, Income),
    social_category(Person, Category),
    income_threshold(Category, Threshold),
    Income > Threshold,
    \+ vulnerable_group(Person, _),
    !.

% Rule N2: Medium income, general category, no vulnerabilities = NOT ELIGIBLE
not_eligible_with_confidence(Person, not_eligible, 0.90) :-
    has_income(Person),
    annual_income(Person, Income),
    Income > 300000,  % Above ₹25k/month
    social_category(Person, general),
    \+ vulnerable_group(Person, _),
    !.

% Rule N3: High income even with reserved category (but no vulnerable status)
not_eligible_with_confidence(Person, not_eligible, 0.85) :-
    has_income(Person),
    annual_income(Person, Income),
    Income > 800000,  % Above ₹8 lakh (max threshold)
    \+ vulnerable_group(Person, _),
    !.

% ============================================================================
% HELPER PREDICATES FOR NEGATIVE ELIGIBILITY CHECKS
% ============================================================================

% Helper: Check if person has high income without vulnerable status
not_eligible_high_income(Person) :-
    social_category(Person, general),
    annual_income(Person, Income),
    Income > 300000,
    no_vulnerable_group(Person).

% Helper: Check if person exceeds category threshold without vulnerable status
not_eligible_exceeds_threshold(Person) :-
    has_income(Person),
    annual_income(Person, Income),
    social_category(Person, Category),
    income_threshold(Category, Threshold),
    Income > Threshold,
    \+ vulnerable_group(Person, _).

% ============================================================================
% ENHANCED ELIGIBILITY CHECK WITH NEGATIVE RULES
% ============================================================================

% Modified main predicate to check negative rules first
eligible_with_confidence_enhanced(Person, Decision, Confidence) :-
    % First check if explicitly NOT eligible
    (   not_eligible_with_confidence(Person, not_eligible, NotEligConf) ->
        Decision = not_eligible,
        Confidence = NotEligConf
    % Then check if eligible
    ;   eligible_with_confidence(Person, eligible, EligConf) ->
        Decision = eligible,
        Confidence = EligConf
    % Default to not eligible with low confidence
    ;   Decision = not_eligible,
        Confidence = 0.60
    ).

% ============================================================================
% INCOME VALIDATION HELPERS
% ============================================================================

% Validate income is in reasonable range
valid_income(Person) :-
    annual_income(Person, Income),
    Income >= 0,
    Income =< 10000000.  % Max ₹10 lakh/year reasonable

% Calculate income category automatically
income_category_auto(Person, Category) :-
    monthly_income(Person, Income),
    (   Income =< 12000 -> Category = very_low
    ;   Income =< 25000 -> Category = low
    ;   Income =< 50000 -> Category = medium
    ;   Category = high
    ).

% ============================================================================
% CONFIDENCE FACTORS
% ============================================================================

% Adjust confidence based on evidence quality
adjust_confidence(BaseConfidence, Person, AdjustedConfidence) :-
    % Start with base confidence
    Conf1 = BaseConfidence,
    
    % Reduce if income is near threshold (uncertain)
    (   near_threshold(Person) ->
        Conf2 is Conf1 * 0.9
    ;   Conf2 = Conf1
    ),
    
    % Increase if multiple eligibility factors
    (   multiple_eligibility_factors(Person) ->
        Conf3 is min(0.98, Conf2 * 1.1)
    ;   Conf3 = Conf2
    ),
    
    AdjustedConfidence = Conf3.

% Check if income is near threshold (±10%)
near_threshold(Person) :-
    has_income(Person),
    annual_income(Person, Income),
    social_category(Person, Category),
    income_threshold(Category, Threshold),
    LowerBound is Threshold * 0.9,
    UpperBound is Threshold * 1.1,
    Income >= LowerBound,
    Income =< UpperBound.

% Check for multiple eligibility factors
multiple_eligibility_factors(Person) :-
    vulnerable_group(Person, _),
    (   no_income(Person)
    ;   has_income(Person),
        annual_income(Person, Income),
        Income < 150000  % Very low income
    ).

% ============================================================================
% END OF LEGAL AID RULES
% ============================================================================
