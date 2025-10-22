% FAMILY LAW DOMAIN RULES - CORRECTED
% Contains marriage, divorce, custody, and maintenance rules
% Corrected to avoid problematic cuts that prevent backtracking

% =================================================================
% MARRIAGE VALIDITY RULES - CORRECTED
% =================================================================

valid_marriage(Person1, Person2, marriage_type(Type)) :-
    marriage_conditions_met(Person1, Person2, Type).

marriage_conditions_met(Person1, Person2, civil) :-
    age_requirement_met(Person1, 21),
    age_requirement_met(Person2, 18).

marriage_conditions_met(Person1, Person2, religious) :-
    age_requirement_met(Person1, 18),
    age_requirement_met(Person2, 18).

age_requirement_met(Person, RequiredAge) :-
    age(Person, ActualAge),
    integer(ActualAge),
    ActualAge >= RequiredAge.

% =================================================================
% DIVORCE GROUNDS - CORRECTED (Removed problematic cuts)
% =================================================================

divorce_grounds_exist(Petitioner, Respondent, Grounds) :-
    valid_divorce_ground(Grounds),
    ground_applicable(Petitioner, Respondent, Grounds).

valid_divorce_ground(cruelty).
valid_divorce_ground(adultery).
valid_divorce_ground(desertion).
valid_divorce_ground(mental_illness).
valid_divorce_ground(conversion).
valid_divorce_ground(renunciation).

ground_applicable(_, _, cruelty).  % Always applicable if proven
ground_applicable(_, _, adultery). % Always applicable if proven
ground_applicable(_, _, desertion) :- !. % Check duration separately
ground_applicable(_, _, mental_illness) :- !. % Medical evidence required
ground_applicable(_, _, conversion) :- !. % Religious conversion
ground_applicable(_, _, renunciation) :- !. % Renunciation of world

% =================================================================
% MAINTENANCE ELIGIBILITY
% =================================================================

maintenance_eligible(Person) :-
    spouse(Person, Spouse),
    (financial_dependency(Person, Spouse) ; children_custody(Person, _)), !.

maintenance_eligible(Person) :-
    parent_child_relationship(Person, Child),
    financial_dependency(Child, Person), !.

% =================================================================
% CHILD CUSTODY PREFERENCES
% =================================================================

child_custody_preference(Child, Parent) :-
    age(Child, Age),
    Age < 7,
    mother(Parent, Child), !.  % Tender years doctrine

child_custody_preference(Child, Parent) :-
    age(Child, Age),
    Age >= 7,
    best_interest_analysis(Child, Parent), !.

best_interest_analysis(Child, Parent) :-
    stable_environment(Parent),
    financial_capability(Parent),
    emotional_bond(Child, Parent), !.

% ...existing code...

family_case_type(Person, divorce) :-
    case_type(Person, divorce), !.
family_case_type(Person, custody) :-
    case_type(Person, custody), !.
family_case_type(Person, maintenance) :-
    case_type(Person, maintenance), !.
family_case_type(Person, domestic_violence) :-
    case_type(Person, domestic_violence), !.

% =================================================================
% UTILITY PREDICATES
% =================================================================

spouse(X, Y) :- married(X, Y).
spouse(X, Y) :- married(Y, X).

mother(Parent, Child) :- parent(Parent, Child), gender(Parent, female).
father(Parent, Child) :- parent(Parent, Child), gender(Parent, male).

% =================================================================
% MISSING UTILITY PREDICATES
% =================================================================

child_custody_case(Person) :-
    case_type(Person, custody).

marriage_validity_case(Person) :-
    case_type(Person, marriage_validity).
