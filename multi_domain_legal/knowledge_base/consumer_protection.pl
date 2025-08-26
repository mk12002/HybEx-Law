% CONSUMER PROTECTION DOMAIN RULES - OPTIMIZED
% Contains consumer forum jurisdiction, complaint handling, and compensation rules
% Optimized for fast consumer case processing

% =================================================================
% CONSUMER FORUM JURISDICTION (Optimized with cuts)
% =================================================================

consumer_forum_jurisdiction(Amount, district_forum) :-
    jurisdiction_limit(district_forum, Limit), Amount =< Limit.

consumer_forum_jurisdiction(Amount, state_commission) :-
    jurisdiction_limit(district_forum, LowerLimit), Amount > LowerLimit,
    jurisdiction_limit(state_commission, UpperLimit), Amount =< UpperLimit.

consumer_forum_jurisdiction(Amount, national_commission) :-
    jurisdiction_limit(state_commission, LowerLimit), Amount > LowerLimit.

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
    compensation_multiplier(Multiplier),
    Compensation is GoodsValue * Multiplier, !.

calculate_base_compensation(defective_goods, Amount) :-
    goods_value(Amount), !.
calculate_base_compensation(deficient_service, Amount) :-
    service_charges_paid(Amount), !.
calculate_base_compensation(_, Amount) :-
    default_minimum_compensation(Amount), !.

calculate_additional_damages(Person, Complaint, Additional) :-
    mental_agony_claimed(Person, Complaint),
    mental_agony_compensation(Additional), !.
calculate_additional_damages(_, _, 0) :- !.  % No additional damages

% ...existing code...

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
