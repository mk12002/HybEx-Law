% ============================================================================
% CROSS-DOMAIN RULES
% This file contains rules that connect multiple legal domains
% Enables comprehensive legal analysis across domain boundaries
% ============================================================================

% =================================================================
% LEGAL AID FOR DOMAIN-SPECIFIC CASES
% =================================================================

% Legal aid for employment matters
legal_aid_employment_case(Employee) :-
    eligible_for_legal_aid(Employee),
    (   wrongful_termination(Employee)
    ;   minimum_wage_violation(Employee)
    ;   valid_harassment_complaint(Employee)
    ).

% Legal aid for family matters
legal_aid_family_case(Person) :-
    eligible_for_legal_aid(Person),
    (   maintenance_eligible(Person)
    ;   divorce_case(Person)
    ;   child_custody_case(Person)
    ).

% Legal aid for consumer matters
legal_aid_consumer_case(Person) :-
    eligible_for_legal_aid(Person),
    valid_consumer_complaint(Person, _),
    transaction_amount(Person, Amount),
    Amount =< 100000.  % Small value cases

% =================================================================
% CONSTITUTIONAL REMEDIES ACROSS DOMAINS
% =================================================================

% Constitutional remedy for employment discrimination
constitutional_employment_remedy(Employee) :-
    employee(Employee),
    discriminated_against(Employee, Grounds),
    prohibited_discrimination_ground(Grounds),
    workplace_discrimination(Employee).

% PIL for employment issues
employment_pil_standing(Person, Issue) :-
    public_interest_issue(Issue),
    employment_related_issue(Issue).

employment_related_issue(widespread_minimum_wage_violation).
employment_related_issue(systemic_workplace_harassment).
employment_related_issue(mass_illegal_terminations).

% =================================================================
% INTER-DOMAIN CASE CLASSIFICATIONS
% =================================================================

% Family law cases that may need constitutional remedy
family_constitutional_case(Person) :-
    family_case(Person),
    fundamental_right_violated(Person, _).

% Consumer cases with employment implications
consumer_employment_case(Person) :-
    valid_consumer_complaint(Person, _),
    employee(Person),
    workplace_related_purchase(Person).

% Employment cases with family implications
employment_family_case(Employee) :-
    wrongful_termination(Employee),
    family_dependent(Employee, _).

% =================================================================
% COMPREHENSIVE ELIGIBILITY ACROSS DOMAINS
% =================================================================

% Legal aid eligibility for any valid case type
comprehensive_legal_aid_eligible(Person) :-
    eligible_for_legal_aid(Person),
    (   legal_aid_employment_case(Person)
    ;   legal_aid_family_case(Person)
    ;   legal_aid_consumer_case(Person)
<<<<<<< HEAD
    ;   criminal_case(Person)
    ;   constitutional_case(Person)
=======
    ;   criminal_case_eligible(Person)
    ;   constitutional_case_eligible(Person)
>>>>>>> f63cb0c5bec52c3c68eb36a972ccaa75026c0afe
    ).

% Multi-domain case complexity
complex_multi_domain_case(Person) :-
    family_case(Person),
    employment_case(Person).

complex_multi_domain_case(Person) :-
    consumer_case(Person),
    constitutional_case(Person).

complex_multi_domain_case(Person) :-
    employment_case(Person),
    constitutional_case(Person).

% =================================================================
% DOMAIN-SPECIFIC CASE TYPE CHECKS
% =================================================================

family_case(Person) :-
    (   divorce_case(Person)
    ;   maintenance_case(Person)
    ;   child_custody_case(Person)
    ;   marriage_validity_case(Person)
    ).

employment_case(Person) :-
    (   wrongful_termination(Person)
    ;   wage_dispute(Person)
    ;   harassment_case(Person)
    ;   discrimination_case(Person)
    ).

consumer_case(Person) :-
    valid_consumer_complaint(Person, _).

constitutional_case(Person) :-
    fundamental_right_violated(Person, _).

criminal_case(Person) :-
    criminal_charges(Person, _).

% =================================================================
% PRIORITY AND URGENCY ASSESSMENT
% =================================================================

urgent_case(Person) :-
    (   domestic_violence_case(Person)
    ;   child_welfare_case(Person)
    ;   immediate_livelihood_threat(Person)
    ;   fundamental_rights_emergency(Person)
    ).

high_priority_case(Person) :-
    (   urgent_case(Person)
    ;   vulnerable_group(Person, _)
    ;   complex_multi_domain_case(Person)
    ).

% =================================================================
% REMEDY RECOMMENDATIONS - IMPROVED LOGIC
% =================================================================

% Constitutional cases always go to high court
recommended_forum(Person, high_court) :-
    constitutional_case(Person).

recommended_forum(Person, high_court) :-
    complex_multi_domain_case(Person).

% Domain-specific forum recommendations (only if not constitutional)
recommended_forum(Person, consumer_forum) :-
    consumer_case(Person),
<<<<<<< HEAD
    \+ constitutional_case(Person).

recommended_forum(Person, family_court) :-
    family_case(Person),
    \+ constitutional_case(Person).

recommended_forum(Person, labor_court) :-
    employment_case(Person),
    \+ constitutional_case(Person).
=======
    constitutional_case(Person, false).

recommended_forum(Person, family_court) :-
    family_case(Person),
    constitutional_case(Person, false).

recommended_forum(Person, labor_court) :-
    employment_case(Person),
    constitutional_case(Person, false).
>>>>>>> f63cb0c5bec52c3c68eb36a972ccaa75026c0afe

% Multiple forum approach for complex cases
multiple_forums_required(Person) :-
    complex_multi_domain_case(Person),
<<<<<<< HEAD
    \+ constitutional_case(Person).
=======
    constitutional_case(Person, false).
>>>>>>> f63cb0c5bec52c3c68eb36a972ccaa75026c0afe

% =================================================================
% COMPREHENSIVE CASE ANALYSIS
% =================================================================

comprehensive_case_analysis(Person, Analysis) :-
    findall(Domain, case_domain(Person, Domain), Domains),
    findall(Forum, recommended_forum(Person, Forum), Forums),
    findall(Priority, case_priority(Person, Priority), Priorities),
    Analysis = analysis(Domains, Forums, Priorities).

case_domain(Person, family) :- family_case(Person).
case_domain(Person, employment) :- employment_case(Person).
case_domain(Person, consumer) :- consumer_case(Person).
case_domain(Person, constitutional) :- constitutional_case(Person).
case_domain(Person, criminal) :- criminal_case(Person).

<<<<<<< HEAD
case_priority(Person, urgent) :- urgent_case(Person), !.
case_priority(Person, high) :- \+ urgent_case(Person), high_priority_case(Person), !.
case_priority(Person, normal) :- \+ urgent_case(Person), \+ high_priority_case(Person).


% ADD THIS TO cross_domain_rules.pl

% Define what a public interest issue is
public_interest_issue(Issue) :-
    employment_related_issue(Issue).
public_interest_issue(widespread_consumer_fraud).
public_interest_issue(environmental_damage).

% Define what constitutes a fundamental rights violation for the case
fundamental_right_violated(Person, right_to_equality) :-
    discriminated_against(Person, Grounds), 
    prohibited_discrimination_ground(Grounds).

fundamental_right_violated(Person, right_to_life) :-
    immediate_livelihood_threat(Person).

prohibited_discrimination_ground(religion).
prohibited_discrimination_ground(race).
prohibited_discrimination_ground(caste).
prohibited_discrimination_ground(sex).
prohibited_discrimination_ground(place_of_birth).
=======
case_priority(Person, urgent) :- urgent_case(Person).
case_priority(Person, high) :- high_priority_case(Person).
% Removed problematic negation - if not urgent or high, Python should explicitly set normal priority
>>>>>>> f63cb0c5bec52c3c68eb36a972ccaa75026c0afe
