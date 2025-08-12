% CONSUMER PROTECTION DOMAIN RULES - OPTIMIZED
% Contains consumer forum jurisdiction, complaint handling, and compensation rules
% Optimized for fast consumer case processing

% =================================================================
% CONSUMER FORUM JURISDICTION (Optimized with cuts)
% =================================================================

consumer_forum_jurisdiction(Amount, district_forum) :-
    Amount =< 2000000, !.  % Up to 20 lakh - District Forum

consumer_forum_jurisdiction(Amount, state_commission) :-
    Amount > 2000000,
    Amount =< 10000000, !.  % 20 lakh to 1 crore - State Commission

consumer_forum_jurisdiction(Amount, national_commission) :-
    Amount > 10000000, !.  % Above 1 crore - National Commission

% =================================================================
% VALID CONSUMER COMPLAINTS (Fast validation)
% =================================================================

valid_consumer_complaint(Person, Complaint) :-
    consumer_issue(Complaint),
    complaint_within_time_limit(Person, Complaint), !.

consumer_issue(defective_goods).
consumer_issue(deficient_service).
consumer_issue(unfair_trade_practice).
consumer_issue(misleading_advertisement).
consumer_issue(overcharging).
consumer_issue(warranty_breach).

complaint_within_time_limit(Person, Complaint) :-
    incident_date(Person, Complaint, IncidentDate),
    complaint_date(Person, Complaint, ComplaintDate),
    days_between(IncidentDate, ComplaintDate, Days),
    Days =< 730, !.  % 2 years time limit

% =================================================================
% CONSUMER COMPENSATION (Optimized calculation)
% =================================================================

consumer_compensation(Person, Complaint, TotalCompensation) :-
    calculate_base_compensation(Complaint, BaseAmount),
    calculate_additional_damages(Person, Complaint, Additional),
    TotalCompensation is BaseAmount + Additional, !.

defective_goods_compensation(GoodsValue, Compensation) :-
    Compensation is GoodsValue * 1.5, !.  % 150% of goods value

calculate_base_compensation(defective_goods, Amount) :-
    goods_value(Amount), !.
calculate_base_compensation(deficient_service, Amount) :-
    service_charges_paid(Amount), !.
calculate_base_compensation(_, 10000) :- !.  % Default minimum

calculate_additional_damages(Person, Complaint, Additional) :-
    mental_agony_claimed(Person, Complaint),
    Additional = 25000, !.
calculate_additional_damages(_, _, 0) :- !.  % No additional damages

% =================================================================
% CONSUMER LEGAL AID
% =================================================================

legal_aid_consumer_case(Person) :-
    eligible_for_legal_aid(Person),  % From legal_aid.pl
    consumer_case_type(Person, _), !.

consumer_case_type(Person, consumer_dispute) :-
    case_type(Person, consumer), !.
consumer_case_type(Person, product_liability) :-
    case_type(Person, product_liability), !.

% =================================================================
% UTILITY PREDICATES
% =================================================================

days_between(Date1, Date2, Days) :-
    % Simplified date calculation
    date_to_days(Date1, Days1),
    date_to_days(Date2, Days2),
    Days is abs(Days2 - Days1), !.

date_to_days(date(Y, M, D), TotalDays) :-
    TotalDays is Y * 365 + M * 30 + D, !.  % Simplified calculation
