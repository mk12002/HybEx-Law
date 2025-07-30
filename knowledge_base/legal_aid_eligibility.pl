% -*- coding: utf-8 -*-
% Prolog Eligibility System for Legal Aid Services (NALSA)
% Based on the Legal Services Authorities Act, 1987
% 
% This knowledge base implements the core eligibility rules for legal aid
% in India, considering financial criteria, categorical eligibility, and
% case type restrictions.

% --- Dynamic facts to be asserted by the NLP pipeline ---
:- dynamic applicant/1.
:- dynamic income_monthly/2.
:- dynamic income_annual/2.
:- dynamic location/2.
:- dynamic case_type/2.
:- dynamic is_woman/2.
:- dynamic is_child/2.
:- dynamic is_sc_st/2.
:- dynamic is_industrial_workman/2.
:- dynamic is_in_custody/2.
:- dynamic is_disabled/2.
:- dynamic is_disaster_victim/2.

% --- Main Eligibility Rule ---
% A person is eligible if they meet EITHER categorical OR financial criteria,
% AND their case type is not explicitly excluded
eligible(Person) :-
    applicant(Person),
    (meets_categorical_criteria(Person) ; meets_financial_criteria(Person)),
    is_eligible_case_type(Person).

% --- Categorical Eligibility (No income limit applies) ---
meets_categorical_criteria(Person) :- 
    is_woman(Person, true).

meets_categorical_criteria(Person) :- 
    is_child(Person, true).

meets_categorical_criteria(Person) :- 
    is_sc_st(Person, true).

meets_categorical_criteria(Person) :- 
    is_industrial_workman(Person, true).

meets_categorical_criteria(Person) :- 
    is_in_custody(Person, true).

meets_categorical_criteria(Person) :- 
    is_disabled(Person, true).

meets_categorical_criteria(Person) :- 
    is_disaster_victim(Person, true).

% --- Financial Eligibility ---
% Income thresholds may vary by state - using general guidelines
meets_financial_criteria(Person) :-
    income_annual(Person, Income),
    Income =< 300000.  % Rs. 3 lakhs per annum (general threshold)

meets_financial_criteria(Person) :-
    income_monthly(Person, Income),
    Income =< 25000.   % Rs. 25,000 per month

% Special case: zero income always qualifies
meets_financial_criteria(Person) :-
    income_monthly(Person, 0).

meets_financial_criteria(Person) :-
    income_annual(Person, 0).

% --- Case Type Eligibility ---
is_eligible_case_type(Person) :-
    case_type(Person, Type),
    \+ is_excluded_case(Type).

% Default: if no case type specified, assume eligible
is_eligible_case_type(Person) :-
    \+ case_type(Person, _).

% --- Excluded Case Types ---
% These case types are generally not eligible for legal aid
is_excluded_case('defamation').
is_excluded_case('malicious_prosecution').
is_excluded_case('economic_offense').
is_excluded_case('election_offense').
is_excluded_case('business_dispute').
is_excluded_case('corporate_matter').

% --- Explanation Predicates ---
% Generate human-readable explanations for eligibility decisions

explanation(Person, eligible, Reason) :-
    eligible(Person),
    generate_eligibility_reason(Person, Reason).

explanation(Person, not_eligible, Reason) :-
    \+ eligible(Person),
    generate_ineligibility_reason(Person, Reason).

generate_eligibility_reason(Person, Reason) :-
    meets_categorical_criteria(Person),
    categorical_reason(Person, Reason).

generate_eligibility_reason(Person, 'Meets financial criteria for legal aid') :-
    meets_financial_criteria(Person),
    \+ meets_categorical_criteria(Person).

categorical_reason(Person, 'Eligible as a woman applicant') :-
    is_woman(Person, true).

categorical_reason(Person, 'Eligible as a child/minor') :-
    is_child(Person, true).

categorical_reason(Person, 'Eligible as SC/ST category member') :-
    is_sc_st(Person, true).

categorical_reason(Person, 'Eligible as an industrial worker') :-
    is_industrial_workman(Person, true).

categorical_reason(Person, 'Eligible as person in custody') :-
    is_in_custody(Person, true).

categorical_reason(Person, 'Eligible as disabled person') :-
    is_disabled(Person, true).

categorical_reason(Person, 'Eligible as disaster victim') :-
    is_disaster_victim(Person, true).

generate_ineligibility_reason(Person, Reason) :-
    case_type(Person, Type),
    is_excluded_case(Type),
    atom_concat('Case type excluded from legal aid: ', Type, Reason).

generate_ineligibility_reason(Person, 'Income exceeds financial eligibility criteria') :-
    \+ meets_financial_criteria(Person),
    \+ meets_categorical_criteria(Person),
    is_eligible_case_type(Person).

generate_ineligibility_reason(Person, 'Does not meet categorical or financial criteria') :-
    \+ meets_financial_criteria(Person),
    \+ meets_categorical_criteria(Person),
    is_eligible_case_type(Person).

% --- Utility Predicates ---
% Check what information is available for a person
available_info(Person, Info) :-
    applicant(Person),
    findall(Fact, 
            (Fact =.. [Predicate, Person, _], 
             call(Fact), 
             Predicate \= applicant), 
            Info).

% Reset all facts for a person (useful for testing)
reset_applicant(Person) :-
    retractall(applicant(Person)),
    retractall(income_monthly(Person, _)),
    retractall(income_annual(Person, _)),
    retractall(location(Person, _)),
    retractall(case_type(Person, _)),
    retractall(is_woman(Person, _)),
    retractall(is_child(Person, _)),
    retractall(is_sc_st(Person, _)),
    retractall(is_industrial_workman(Person, _)),
    retractall(is_in_custody(Person, _)),
    retractall(is_disabled(Person, _)),
    retractall(is_disaster_victim(Person, _)).

% --- Sample Test Cases ---
% These can be used for testing the knowledge base

test_case_1 :-
    % Low income general category person
    assertz(applicant(person1)),
    assertz(income_monthly(person1, 15000)),
    assertz(case_type(person1, 'property_dispute')),
    assertz(is_woman(person1, false)),
    assertz(is_sc_st(person1, false)).

test_case_2 :-
    % Woman applicant (categorical eligibility)
    assertz(applicant(person2)),
    assertz(income_monthly(person2, 50000)),
    assertz(case_type(person2, 'domestic_violence')),
    assertz(is_woman(person2, true)).

test_case_3 :-
    % High income with excluded case type
    assertz(applicant(person3)),
    assertz(income_monthly(person3, 100000)),
    assertz(case_type(person3, 'defamation')),
    assertz(is_woman(person3, false)),
    assertz(is_sc_st(person3, false)).
