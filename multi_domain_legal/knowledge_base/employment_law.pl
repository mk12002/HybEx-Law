

% =================================================================
% WRONGFUL TERMINATION - CORRECTED
% =================================================================

wrongful_termination(Employee) :-
    improper_procedure(Employee).
wrongful_termination(Employee) :-
    insufficient_notice(Employee).
wrongful_termination(Employee) :-
    discriminatory_termination(Employee).

% Corrected discriminatory_termination to avoid circular dependency
discriminatory_termination(Employee) :-
    discriminated_against(Employee, _),
    case_type(Employee, discrimination).
    
improper_procedure(Employee) :-
    \+ disciplinary_hearing_conducted(Employee).
improper_procedure(Employee) :-
    \+ opportunity_to_explain_given(Employee).

insufficient_notice(Employee) :-
    notice_period_given(Employee, GivenPeriod),
    required_notice_period(Employee, RequiredPeriod),
    integer(GivenPeriod),
    integer(RequiredPeriod),
    GivenPeriod < RequiredPeriod.

sufficient_notice_period(Employee, GivenPeriod) :-
    required_notice_period(Employee, RequiredPeriod),
    integer(GivenPeriod),
    integer(RequiredPeriod),
    GivenPeriod >= RequiredPeriod.

required_notice_period(Employee, Period) :-
    employment_duration(Employee, Duration),
    notice_period_for_duration(Duration, Period).

% =================================================================
% RETRENCHMENT COMPENSATION
% =================================================================

retrenchment_compensation(Employee, Amount) :-
    employment_duration(Employee, Days),
    daily_wage(Employee, Wage),
    Amount is (Days / 365) * 15 * Wage, !.  % 15 days wage per year

% =================================================================
% WORKPLACE HARASSMENT
% =================================================================

valid_harassment_complaint(Employee) :-
    harassment_incident_reported(Employee),
    reported_within_time_limit(Employee), !.

harassment_remedy_available(Employee, Remedy) :-
    valid_harassment_complaint(Employee),
    harassment_remedy(Remedy), !.

harassment_remedy(compensation).
harassment_remedy(transfer).
harassment_remedy(disciplinary_action).
harassment_remedy(awareness_training).

% =================================================================
% MINIMUM WAGE VIOLATIONS
% =================================================================

minimum_wage_violation(Employee) :-
    actual_wage(Employee, ActualWage),
    minimum_wage_rate(Employee, MinWage),
    ActualWage < MinWage, !.

minimum_wage_rate(Employee, Rate) :-
    employment_state(Employee, State),
    state_minimum_wage(State, Rate), !.
minimum_wage_rate(_, 176) :- !.  % Default national minimum wage

overtime_payment_due(Employee, Amount) :-
    overtime_hours(Employee, Hours),
    Hours > 0,
    hourly_wage(Employee, Wage),
    Amount is Hours * Wage * 2, !.  % Double rate for overtime

overtime_hours(Employee, OvertimeHours) :-
    daily_working_hours(Employee, DailyHours),
    DailyHours > 8,
    OvertimeHours is DailyHours - 8, !.
overtime_hours(_, 0) :- !.  % No overtime

% =================================================================
% EMPLOYMENT LEGAL AID
% =================================================================

% ...existing code...

employment_case_type(Person, wrongful_termination) :-
    case_type(Person, employment),
    wrongful_termination(Person), !.
employment_case_type(Person, wage_dispute) :-
    case_type(Person, wage_dispute), !.
employment_case_type(Person, harassment) :-
    case_type(Person, harassment), !.

% =================================================================
% CONSTITUTIONAL EMPLOYMENT REMEDIES
% =================================================================

% =================================================================
% MISSING UTILITY PREDICATES
% =================================================================

notice_period_for_duration(Days, 30) :- Days < 365, !.
notice_period_for_duration(Days, 60) :- Days < 1825, !.  % < 5 years
notice_period_for_duration(_, 90).  % >= 5 years

state_minimum_wage('maharashtra', 18000).
state_minimum_wage('karnataka', 16000).
state_minimum_wage('delhi', 17000).
state_minimum_wage(_, 15000).  % Default


