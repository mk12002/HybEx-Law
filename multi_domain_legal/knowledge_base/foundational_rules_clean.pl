% ============================================================================
% FOUNDATIONAL PREDICATES AND UTILITY RULES - CORRECTED AND ROBUST
% This file contains only generic, reusable predicates.
% All case-specific facts (e.g., age, income) must be provided by Python.
% ============================================================================

% Simple utility predicate to check if an element is in a list
member(X, [X|_]).
member(X, [_|T]) :- member(X, T).

% Basic predicate for any valid person identifier
person(Person) :- atom(Person).

% =================================================================
% VULNERABLE GROUP DEFINITIONS - CORRECTED LOGIC
% =================================================================

% These rules are deterministic and should not cause infinite loops
vulnerable_group(Person, 'women') :-
    gender(Person, 'female').

vulnerable_group(Person, 'children') :-
    age(Person, Age),
    integer(Age),
    Age < 18.

vulnerable_group(Person, 'disabled') :-
    disability_status(Person, true).

vulnerable_group(Person, 'senior_citizens') :-
    age(Person, Age),
    integer(Age),
    Age >= 60.

vulnerable_group(Person, 'industrial_workers') :-
    occupation(Person, 'industrial_worker').

% =================================================================
% UTILITY PREDICATES - SAFE DEFAULT DEFINITIONS
% =================================================================
% Catch-all rules to prevent "Unknown procedure" errors.
% These simply fail gracefully if no other rule or fact exists.
disability_status(_, false).
occupation(_, unknown).
gender(_, unknown).
