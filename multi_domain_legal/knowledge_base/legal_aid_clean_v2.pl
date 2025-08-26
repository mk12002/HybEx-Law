% =================================================================
% LEGAL AID DOMAIN RULES - CORRECTED AND ROBUST
% =================================================================

% A person is eligible for legal aid if they meet ANY of the criteria.
% This is the main, top-level rule using logical disjunctions.
eligible_for_legal_aid(Person) :-
    (   categorically_eligible(Person)     % Checks SC, ST, OBC, BPL
    ;   vulnerable_group_eligible(Person)  % Checks women, children, senior citizens, disabled, etc.
    ;   income_eligible(Person)           % Checks if income is below threshold
    ).

% =================================================================
% CATEGORICAL ELIGIBILITY (Most definitive)
% =================================================================

categorically_eligible(Person) :- social_category(Person, 'sc').
categorically_eligible(Person) :- social_category(Person, 'st').
categorically_eligible(Person) :- social_category(Person, 'obc').
categorically_eligible(Person) :- social_category(Person, 'bpl').

% =================================================================
% VULNERABLE GROUP ELIGIBILITY
% =================================================================
% This predicate reuses the logic from foundational_rules_clean.pl
vulnerable_group_eligible(Person) :- vulnerable_group(Person, _).

% =================================================================
% STATIC INCOME THRESHOLDS (Values from HybExConfig)
% Adding these here makes the KB self-contained for generation.
% =================================================================

income_threshold('sc', 800000).
income_threshold('st', 800000).
income_threshold('obc', 600000).
income_threshold('bpl', 0). % BPL is categorically eligible, but a threshold is good practice.
income_threshold('ews', 800000).
income_threshold('general', 500000).

% =================================================================
% INCOME ELIGIBILITY (Dynamic thresholds from Python)
% =================================================================
% Removed cuts for better logical flow
income_eligible(Person) :-
    annual_income(Person, Income),
    social_category(Person, Category),
    income_threshold(Category, Threshold),
    Income =< Threshold.

% Fallback for cases with no social category
income_eligible(Person) :-
    annual_income(Person, Income),
    \+ social_category(Person, _),
    income_threshold('general', Threshold),
    Income =< Threshold.

% =================================================================
% REASONING PREDICATES (Corrected and optimized with cuts for speed)
% =================================================================

% Primary reasons in order of precedence (cuts ensure only one reason is returned)
primary_eligibility_reason(Person, 'Eligible due to categorical status (SC/ST/OBC/BPL).') :-
    categorically_eligible(Person), !.

primary_eligibility_reason(Person, 'Eligible due to vulnerable group status.') :-
    vulnerable_group_eligible(Person), !.

primary_eligibility_reason(Person, 'Eligible due to income below threshold.') :-
    income_eligible(Person), !.

primary_eligibility_reason(Person, 'Not eligible based on current criteria.') :-
    \+ eligible_for_legal_aid(Person), !.

% =================================================================
% DETAILED REASONING GENERATION - ENHANCED
% =================================================================

generate_detailed_reasoning(Person, Detailed) :-
    primary_eligibility_reason(Person, Primary),
    (   categorically_eligible(Person) ->
        Extra = ' Categorical eligibility under Legal Services Authorities Act, 1987 for SC/ST/OBC/BPL categories, regardless of income.'
    ;   vulnerable_group_eligible(Person) ->
        Extra = ' Vulnerable group eligibility under Section 12 of Legal Services Authorities Act for women, children, seniors, disabled, etc.'
    ;   income_eligible(Person) ->
        Extra = ' Income-based eligibility as annual income is below the applicable state threshold (varies by category).'
    ;   Extra = ' No additional eligibility details available.'
    ),
    atom_concat(Primary, Extra, Detailed).

% =================================================================
% APPLICABLE RULES IDENTIFICATION
% =================================================================

applicable_rule(Person, 'categorical_eligibility', 'legal_aid') :-
    categorically_eligible(Person).

applicable_rule(Person, 'vulnerable_group_eligibility', 'legal_aid') :-
    vulnerable_group_eligible(Person).

applicable_rule(Person, 'income_based_eligibility', 'legal_aid') :-
    income_eligible(Person).
