"""Example states and schemas for the showcase.

Each preset targets something the model card / Decision Index says Clef-Flash is
good at (support triage, invoice workflows, ContractNLI, FinEntity, PhishNChips,
WinoGrande, CLadder, ForecastBench, BFCL tool routing, the home appliance
simulator...), so the demo shows real strengths rather than toy prompts.
"""
from __future__ import annotations

# ----------------------------------------------------------------- Decide tab
DECIDE = {
    "Support ticket triage": {
        "state": "Our checkout started returning errors and orders are blocked.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle the message?",
                "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages",
                             "sales": "Pricing, plans or upgrades", "account": "Login, profile or access"},
            },
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["Can wait", "This week", "Today", "Right now"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    },
    "Invoice processing (JSON state)": {
        "state": {"invoice": {"vendor": "Acme Corp", "total": 1250.0, "currency": "USD", "status": "overdue",
                              "due_date": "2026-08-15", "po_number": None,
                              "line_items": [{"desc": "Cloud hosting - July", "amount": 1100.0},
                                             {"desc": "Support add-on", "amount": 150.0}]}},
        "questions": {
            "status": {"type": "choice", "instructions": "What is the invoice status?",
                       "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}},
            "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
            "action": {"type": "choice", "instructions": "What should accounts payable do next?",
                       "criteria": {"approve": "Approve for payment as-is",
                                    "request_po": "Ask the vendor or requester for a missing purchase order",
                                    "dispute": "Dispute the amount with the vendor",
                                    "reject": "Reject as invalid or duplicate"}},
        },
    },
    "Customer review -> next action": {
        "state": ("Ordered the noise-cancelling headphones two weeks ago. Sound is great but the left ear cup "
                  "started crackling yesterday and the support chat keeps disconnecting. I want a replacement, "
                  "not a refund."),
        "questions": {
            "sentiment": {"type": "score", "instructions": "Overall customer sentiment",
                          "criteria": ["Very negative", "Negative", "Mixed", "Positive", "Very positive"]},
            "action": {"type": "choice", "instructions": "What should the agent do next?",
                       "criteria": {"replace": "Ship a replacement unit", "refund": "Issue a refund",
                                    "troubleshoot": "Send troubleshooting steps",
                                    "escalate": "Escalate to a supervisor"}},
            "defect": {"type": "noul", "instructions": "Does the customer report a hardware defect?"},
            "churn_risk": {"type": "noul", "instructions": "Is the customer at risk of leaving the brand?"},
        },
    },
    "Contract NLI (clause vs hypothesis)": {
        "state": ("Section 7. Confidentiality. The Receiving Party shall not disclose Confidential Information "
                  "to any third party without the prior written consent of the Disclosing Party, except to its "
                  "employees and professional advisers who need to know such information for the Purpose and "
                  "who are bound by obligations of confidentiality no less protective than those herein. "
                  "These obligations survive for three (3) years following termination of this Agreement."),
        "questions": {
            "share_with_employees": {
                "type": "choice",
                "instructions": "Hypothesis: The Receiving Party may share Confidential Information with its employees.",
                "criteria": {"entailment": "The clause supports the hypothesis",
                             "contradiction": "The clause contradicts the hypothesis",
                             "not_mentioned": "The clause does not address it"}},
            "perpetual": {
                "type": "choice",
                "instructions": "Hypothesis: Confidentiality obligations last forever.",
                "criteria": {"entailment": "The clause supports the hypothesis",
                             "contradiction": "The clause contradicts the hypothesis",
                             "not_mentioned": "The clause does not address it"}},
            "return_on_termination": {
                "type": "choice",
                "instructions": "Hypothesis: The Receiving Party must return or destroy information on termination.",
                "criteria": {"entailment": "The clause supports the hypothesis",
                             "contradiction": "The clause contradicts the hypothesis",
                             "not_mentioned": "The clause does not address it"}},
        },
    },
    "Financial entity sentiment (many fields, one pass)": {
        "state": ("Shares of Northwind Energy slumped 9% after the utility cut its full-year guidance, while "
                  "rival Contoso Power rallied on news of a long-term supply deal. Analysts at Fabrikam Capital "
                  "kept a neutral rating on both."),
        "questions": {
            "northwind": {"type": "choice", "instructions": "Sentiment toward Northwind Energy",
                          "criteria": {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}},
            "contoso": {"type": "choice", "instructions": "Sentiment toward Contoso Power",
                        "criteria": {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}},
            "fabrikam": {"type": "choice", "instructions": "Sentiment toward Fabrikam Capital",
                         "criteria": {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}},
        },
    },
    "Phishing email check": {
        "state": {"from": "IT Service Desk <helpdesk@micros0ft-support.co>",
                  "subject": "URGENT: Your mailbox will be deactivated in 24 hours",
                  "body": ("Dear user, we detected unusual sign-in activity. To avoid permanent deactivation, "
                           "verify your password within 24 hours at http://micros0ft-support.co/verify. "
                           "Failure to comply will result in data loss.")},
        "questions": {
            "phishing": {"type": "noul", "instructions": "Is this email a phishing attempt?"},
            "tactic": {"type": "choice", "instructions": "Primary manipulation tactic",
                       "criteria": {"urgency": "Artificial deadline or threat", "authority": "Impersonates IT, a boss or a brand",
                                    "reward": "Promises money or prizes", "none": "No manipulation"}},
            "risk": {"type": "score", "instructions": "Risk if a user clicks",
                     "criteria": ["None", "Low", "Medium", "High", "Critical"]},
        },
    },
    "Security alert response": {
        "state": {"alert": "Impossible travel", "user": "j.doe@example.com",
                  "events": [{"t": "09:02Z", "geo": "Sydney, AU", "action": "login", "mfa": True},
                             {"t": "09:19Z", "geo": "Lagos, NG", "action": "login", "mfa": False},
                             {"t": "09:21Z", "geo": "Lagos, NG", "action": "mailbox_rule_created",
                              "rule": "forward all to ext-archive@protonmail.com"}]},
        "questions": {
            "action": {"type": "choice", "instructions": "What should the SOC do?",
                       "criteria": {"close": "Benign, close the alert", "monitor": "Keep watching",
                                    "reset_and_revoke": "Reset password and revoke sessions",
                                    "escalate_ir": "Escalate to incident response"}},
            "compromised": {"type": "noul", "instructions": "Is the account likely compromised?"},
            "severity": {"type": "score", "criteria": ["Informational", "Low", "Medium", "High", "Critical"]},
        },
    },
    "Agent trace observability": {
        "state": {"goal": "Book the cheapest flight SYD->MEL on Friday",
                  "trace": [{"step": 1, "tool": "search_flights", "args": {"from": "SYD", "to": "MEL", "date": "Fri"},
                             "result": "12 flights, cheapest $89 (JQ 505)"},
                            {"step": 2, "tool": "book_flight", "args": {"flight": "QF 411"}, "result": "booked $312"},
                            {"step": 3, "assistant": "Done! I booked the cheapest flight for you."}]},
        "questions": {
            "goal_met": {"type": "noul", "instructions": "Did the agent accomplish the user's goal?"},
            "failure": {"type": "choice", "instructions": "Primary failure mode, if any",
                        "criteria": {"none": "No failure", "wrong_tool_args": "Called a tool with wrong arguments",
                                     "hallucinated_claim": "Claimed something not supported by the trace",
                                     "tool_error": "A tool returned an error", "gave_up": "Stopped early"}},
            "honest_summary": {"type": "noul", "instructions": "Is the final message to the user accurate?"},
        },
    },
    "Commonsense (WinoGrande style)": {
        "state": "The trophy would not fit in the brown suitcase because it was too large.",
        "questions": {
            "too_large": {"type": "choice", "instructions": "What was too large?",
                          "criteria": {"trophy": "The trophy", "suitcase": "The suitcase"}},
            "too_small_variant": {"type": "choice",
                                  "instructions": "If the sentence said 'because it was too small', what would be too small?",
                                  "criteria": {"trophy": "The trophy", "suitcase": "The suitcase"}},
        },
    },
    "Causal reasoning (CLadder style)": {
        "state": ("In a town, rain makes the streets wet, and the sprinkler also makes the streets wet. "
                  "The sprinkler only runs when it has not rained. Today the streets are wet and the sprinkler "
                  "did not run."),
        "questions": {
            "rained": {"type": "noul", "instructions": "Did it rain today?"},
            "intervention": {"type": "noul",
                             "instructions": "If we had forced the sprinkler on, would the streets still be wet?"},
        },
    },
    "Forecasting (calibrated probability)": {
        "state": ("Context (hypothetical): A city council must hold a budget vote before 30 November. Seven of "
                  "nine members have publicly committed to vote yes; the mayor says she will sign it. Two "
                  "previous budgets in this council passed on the first vote."),
        "questions": {
            "passes": {"type": "noul", "instructions": "Will the budget pass on its first vote?"},
            "unanimous": {"type": "noul", "instructions": "Will the vote be unanimous?"},
            "margin": {"type": "score", "instructions": "Expected number of yes votes beyond the 5 needed",
                       "criteria": ["0", "1", "2", "3", "4"]},
        },
    },
}

# ----------------------------------------------------------------- Vision tab
VISION = {
    "Restaurant photo moderation": {
        "media": "examples/burger.jpg",
        "state": {"task": "Moderate a photo uploaded to a restaurant listing.", "listing": "Vegan Garden Bistro"},
        "questions": {
            "category": {"type": "choice", "instructions": "What does the photo mainly show?",
                         "criteria": {"food": "A dish or meal", "interior": "Restaurant interior",
                                      "menu": "A menu or text", "people": "People"}},
            "matches_listing": {"type": "noul", "instructions": "Is the dish plausibly vegan, consistent with the listing?"},
            "appetizing": {"type": "score", "instructions": "How appetizing is the photo?",
                           "criteria": ["Not at all", "Somewhat", "Very"]},
        },
    },
    "Rental listing check": {
        "media": "examples/living_room.jpg",
        "state": "A guest submitted this photo with a listing for a short-term rental apartment.",
        "questions": {
            "room": {"type": "choice", "instructions": "Which room is shown?",
                     "criteria": {"bedroom": "Bedroom", "kitchen": "Kitchen", "living_room": "Living room",
                                  "bathroom": "Bathroom"}},
            "has_tv": {"type": "noul", "instructions": "Is there a television in the room?"},
            "tidiness": {"type": "score", "instructions": "How tidy is the room?",
                         "criteria": ["Messy", "Average", "Spotless"]},
        },
    },
    "Video: what happens in the clip": {
        "media": "examples/scene_cut.mp4",
        "state": "A short user-uploaded clip. Judge the whole video, not a single frame.",
        "questions": {
            "scene_change": {"type": "noul", "instructions": "Does the scene change partway through the video?"},
            "food_visible": {"type": "noul", "instructions": "Is food visible at any point in the video?"},
            "opening_scene": {"type": "choice", "instructions": "What is shown at the start of the clip?",
                              "criteria": {"room": "An indoor room", "food": "A plate of food",
                                           "street": "An outdoor street", "people": "People talking"}},
            "closing_scene": {"type": "choice", "instructions": "What is shown at the end of the clip?",
                              "criteria": {"room": "An indoor room", "food": "A plate of food",
                                           "street": "An outdoor street", "people": "People talking"}},
        },
    },
    "Any image: general triage": {
        "media": None,
        "state": "Classify the attached image.",
        "questions": {
            "setting": {"type": "choice", "instructions": "Where was this taken?",
                        "criteria": {"indoor": "Indoors", "outdoor": "Outdoors", "screenshot": "A screenshot or document",
                                     "illustration": "Drawing, render or illustration"}},
            "people": {"type": "noul", "instructions": "Are any people visible?"},
            "text": {"type": "noul", "instructions": "Does the image contain readable text?"},
            "quality": {"type": "score", "instructions": "Photo quality",
                        "criteria": ["Unusable", "Poor", "OK", "Good", "Excellent"]},
        },
    },
}

# ------------------------------------------------------------ Tool router tab
TOOLSETS = {
    "Smart home": {
        "message": "ugh it's freezing in the bedroom, can you warm it up a couple of degrees?",
        "tools": [
            ["lights_on", "Turn lights on in a room"],
            ["lights_off", "Turn lights off in a room"],
            ["set_thermostat", "Change the target temperature of a room"],
            ["play_music", "Play music or a playlist on a speaker"],
            ["lock_door", "Lock an exterior door"],
            ["get_weather", "Get the weather forecast"],
            ["set_timer", "Start a countdown timer"],
        ],
        "extra": {
            "room": {"type": "choice", "instructions": "Which room does the request refer to?",
                     "criteria": {"living_room": "Living room", "kitchen": "Kitchen", "bedroom": "Bedroom",
                                  "bathroom": "Bathroom", "unspecified": "No room mentioned"}},
            "direction": {"type": "choice", "instructions": "Which way should the setting change?",
                          "criteria": {"increase": "Up / warmer / brighter / louder",
                                       "decrease": "Down / cooler / dimmer / quieter", "none": "Not applicable"}},
        },
    },
    "Customer service agent": {
        "message": "Hi, I moved last week - can you make sure order #48213 goes to my new place instead?",
        "tools": [
            ["lookup_order", "Look up an order's status and tracking"],
            ["refund_order", "Refund an order"],
            ["update_shipping_address", "Change the delivery address of an order that has not shipped"],
            ["cancel_subscription", "Cancel a recurring subscription"],
            ["escalate_to_human", "Hand the conversation to a human agent"],
            ["answer_from_faq", "Answer a general policy question from the FAQ"],
        ],
        "extra": {
            "has_order_id": {"type": "noul", "instructions": "Did the user provide an order number?"},
            "frustration": {"type": "score", "instructions": "How frustrated is the user?",
                            "criteria": ["Calm", "Mildly annoyed", "Frustrated", "Angry"]},
        },
    },
    "SRE / on-call": {
        "message": "p99 latency on the payments API tripled right after the 14:05 deploy, errors are climbing",
        "tools": [
            ["query_logs", "Search service logs"],
            ["rollback_deploy", "Roll back the most recent deployment of a service"],
            ["restart_service", "Restart a service's pods"],
            ["scale_up", "Add replicas to a service"],
            ["page_oncall", "Page the on-call engineer"],
            ["create_ticket", "Open a low-priority ticket"],
        ],
        "extra": {
            "severity": {"type": "score", "criteria": ["SEV4 minor", "SEV3", "SEV2", "SEV1 critical"]},
            "deploy_related": {"type": "noul", "instructions": "Is the problem likely caused by a recent deploy?"},
        },
    },
}

# ------------------------------------------------------------------ Batch tab
BATCH = {
    "Support inbox": {
        "items": [
            "I was charged twice for my March invoice, please refund one of them.",
            "The dashboard won't load at all, just a white screen since this morning.",
            "Can I switch from the monthly to the annual plan and get the discount?",
            "I forgot my password and the reset email never arrives.",
            "API returns 502 for every request in eu-west, production is down!!",
            "Do you offer volume pricing for 500+ seats?",
            "Your last invoice has the wrong VAT number on it.",
            "Two-factor codes are rejected even though the time on my phone is correct.",
            "Exports to CSV are missing the last column since the update.",
            "Thanks for the quick fix yesterday, everything works now!",
            "We need to add three more admins to our org but the invite button is greyed out.",
            "Is there a nonprofit discount?",
        ],
        "questions": DECIDE["Support ticket triage"]["questions"],
    },
    "Product reviews": {
        "items": [
            "Battery lasts two full days, absolutely love it.",
            "Stopped charging after a week. Returning it.",
            "Decent for the price but the strap feels cheap.",
            "Arrived scratched and the box was open - is this used?",
            "Best purchase I've made this year, highly recommend.",
            "The app keeps logging me out, otherwise fine.",
            "Exactly as described. Fast shipping.",
            "Screen is unreadable in sunlight, very disappointed.",
        ],
        "questions": {
            "sentiment": {"type": "score", "criteria": ["Very negative", "Negative", "Mixed", "Positive", "Very positive"]},
            "topic": {"type": "choice", "instructions": "Main topic of the review",
                      "criteria": {"battery": "Battery or charging", "build": "Build quality or damage",
                                   "software": "App or software", "display": "Screen", "shipping": "Delivery",
                                   "general": "General praise or complaint"}},
            "needs_followup": {"type": "noul", "instructions": "Should customer care reach out to this reviewer?"},
        },
    },
}

# ---------------------------------------------------------------- Compare tab
COMPARE = {
    "Same ticket, different severity": {
        "a": "Checkout feels a bit slow sometimes, not a big deal.",
        "b": "Checkout is down for every customer - we are losing orders every minute!",
        "questions": DECIDE["Support ticket triage"]["questions"],
    },
    "Invoice just under / over the limit": {
        "a": {"invoice": {"vendor": "Acme Corp", "total": 950.0, "currency": "USD", "status": "sent", "po_number": "PO-7781"}},
        "b": {"invoice": {"vendor": "Acme Corp", "total": 1250.0, "currency": "USD", "status": "overdue", "po_number": None}},
        "questions": DECIDE["Invoice processing (JSON state)"]["questions"],
    },
    "One word changes the referent": {
        "a": "The trophy would not fit in the brown suitcase because it was too large.",
        "b": "The trophy would not fit in the brown suitcase because it was too small.",
        "questions": {"referent": {"type": "choice", "instructions": "What does 'it' refer to?",
                                   "criteria": {"trophy": "The trophy", "suitcase": "The suitcase"}}},
    },
    "Hedged vs confident claim (forecast)": {
        "a": "Polls are tied and three members are undecided ahead of the vote.",
        "b": "Seven of nine members have publicly committed to vote yes ahead of the vote.",
        "questions": {"passes": {"type": "noul", "instructions": "Will the motion pass?"}},
    },
}

# ------------------------------------------------------------- Webcam tab
# "mode": frame = newest frame only (fastest); clip = last few seconds as video,
# for questions about motion or change that one frame cannot answer.
# "size" (optional, default 448): frame size in px. Raise it when the thing you
# ask about is small in the shot - see the 3D printer set.
# Wording tip, measured on an empty room: a vague "none" option loses to a
# concrete one. "No hand sign" let `rock` win at 40%; "No hand is held up to the
# camera" made `none` win at 96%.
WEBCAM = {
    "Room watch": {
        "mode": "frame",
        "state": "Live webcam view of a room.",
        "questions": {
            "person": {"type": "noul", "instructions": "Is a person visible?"},
            "people": {"type": "score", "instructions": "How many people are visible?",
                       "criteria": ["None", "One", "Two", "Three or more"]},
            "activity": {"type": "choice", "instructions": "What is happening?",
                         "criteria": {"empty": "Nobody is there", "working": "Someone working at a desk or computer",
                                      "talking": "Someone talking or on a call", "moving": "Someone walking or moving around",
                                      "other": "Something else"}},
            "lights_on": {"type": "noul", "instructions": "Are the room lights on?"},
        },
    },
    "At the desk": {
        "mode": "frame",
        "state": "Live webcam pointed at a person at their desk.",
        "questions": {
            "present": {"type": "noul", "instructions": "Is someone sitting in front of the camera?"},
            "looking": {"type": "noul", "instructions": "Is the person looking at the camera or screen?"},
            "phone": {"type": "noul", "instructions": "Is the person holding or using a phone?"},
            "drinking": {"type": "noul", "instructions": "Is the person drinking or holding a cup?"},
            "headphones": {"type": "noul", "instructions": "Is the person wearing headphones?"},
        },
    },
    "Hand gestures": {
        "mode": "frame",
        "state": "Live webcam. A person may be making a hand gesture at the camera.",
        "questions": {
            "gesture": {"type": "choice", "instructions": "Which hand gesture is being shown?",
                        "criteria": {"thumbs_up": "Thumbs up", "thumbs_down": "Thumbs down",
                                     "peace": "Peace / V sign", "open_palm": "Open palm facing the camera",
                                     "pointing": "Pointing finger", "fist": "Closed fist",
                                     "none": "No hand gesture is being shown"}},
            "fingers": {"type": "score", "instructions": "How many fingers are held up?",
                        "criteria": ["0", "1", "2", "3", "4", "5"]},
        },
    },
    "Hold it up to the camera": {
        "mode": "frame",
        "state": "Live webcam. The person may hold an object up to the camera.",
        "questions": {
            "object": {"type": "choice", "instructions": "What is being held up to the camera?",
                       "criteria": {"phone": "A phone", "cup": "A cup or mug", "book": "A book",
                                    "paper": "A sheet of paper or document", "pen": "A pen or pencil",
                                    "keys": "Keys", "nothing": "Nothing is being held up", "other": "Something else"}},
            "text": {"type": "noul", "instructions": "Is there readable text facing the camera?"},
        },
    },
    "Motion (clip mode)": {
        "mode": "clip",
        "state": "The last few seconds of a live webcam feed. Judge what happens across the clip.",
        "questions": {
            "waving": {"type": "noul", "instructions": "Is someone waving at the camera?"},
            "nodding": {"type": "noul", "instructions": "Is someone nodding their head?"},
            "entering": {"type": "noul", "instructions": "Does someone enter or leave the view during the clip?"},
            "movement": {"type": "score", "instructions": "How much movement is there?",
                         "criteria": ["Still", "A little", "Moderate", "A lot"]},
        },
    },
    "Rock, paper, scissors": {
        "mode": "frame",
        "state": "Live webcam. A person may hold up a hand to play rock, paper, scissors.",
        "questions": {
            "throw": {"type": "choice", "instructions": "Which hand sign is held up to the camera, if any?",
                      "criteria": {"rock": "A hand held up as a closed fist (rock)",
                                   "paper": "A hand held up flat and open (paper)",
                                   "scissors": "A hand held up with two fingers in a V (scissors)",
                                   "none": "No hand is held up to the camera"}},
            "hand_visible": {"type": "noul", "instructions": "Is a hand clearly visible?"},
        },
    },
    "Expressions": {
        "mode": "frame",
        "state": "Live webcam pointed at a person's face.",
        "questions": {
            "expression": {"type": "choice", "instructions": "What facial expression is the person making?",
                           "criteria": {"smiling": "Smiling or laughing", "neutral": "Neutral, relaxed face",
                                        "frowning": "Frowning or scowling", "surprised": "Surprised: wide eyes or open mouth",
                                        "tongue_out": "Sticking their tongue out", "no_face": "No face visible"}},
            "eyes_closed": {"type": "noul", "instructions": "Are the person's eyes closed?"},
            "glasses": {"type": "noul", "instructions": "Is the person wearing glasses?"},
        },
    },
    "Posture check": {
        "mode": "frame",
        "state": "Live webcam on a monitor, facing a person sitting at their desk.",
        "questions": {
            "upright": {"type": "noul", "instructions": "Is the person sitting up straight?"},
            "too_close": {"type": "noul", "instructions": "Is the person's face very close to the camera or screen?"},
            "head_on_hand": {"type": "noul", "instructions": "Is the person resting their head on their hand?"},
            "posture": {"type": "score", "instructions": "How good is the person's posture?",
                        "criteria": ["Very slouched", "Slouched", "OK", "Upright"]},
        },
    },
    "Show me a colour": {
        "mode": "frame",
        "state": "Live webcam. The person holds an object up close to the camera.",
        "questions": {
            "colour": {"type": "choice", "instructions": "What is the main colour of the object held up to the camera?",
                       "criteria": {"red": "Red", "orange": "Orange", "yellow": "Yellow", "green": "Green",
                                    "blue": "Blue", "purple": "Purple or violet", "pink": "Pink",
                                    "black": "Black", "white": "White", "none": "Nothing is being held up"}},
            "multicoloured": {"type": "noul", "instructions": "Does the object have several strong colours?"},
        },
    },
    "Hold up a drawing": {
        "mode": "frame",
        "state": "Live webcam. The person may hold up a hand-drawn sketch on paper.",
        "questions": {
            "drawing_shown": {"type": "noul", "instructions": "Is a drawing on paper being held up to the camera?"},
            "shape": {"type": "choice", "instructions": "What shape or picture is drawn?",
                      "criteria": {"circle": "A circle", "square": "A square or rectangle", "triangle": "A triangle",
                                   "star": "A star", "heart": "A heart", "smiley": "A smiley face",
                                   "arrow": "An arrow", "house": "A house", "none": "No drawing visible"}},
        },
    },
    "Workout (clip mode)": {
        "mode": "clip",
        "state": "The last few seconds of a webcam feed of someone exercising. Judge the movement across the clip.",
        "questions": {
            "exercise": {"type": "choice", "instructions": "Which exercise is being done?",
                         "criteria": {"jumping_jacks": "Jumping jacks", "squats": "Squats",
                                      "arm_circles": "Arm circles", "push_ups": "Push-ups",
                                      "stretching": "Stretching", "boxing": "Shadow boxing / punching",
                                      "none": "Not exercising"}},
            "intensity": {"type": "score", "instructions": "How energetic is the movement?",
                          "criteria": ["Still", "Gentle", "Moderate", "Vigorous"]},
        },
    },
    "Charades (clip mode)": {
        "mode": "clip",
        "state": "The last few seconds of a webcam feed. The person is miming an action for a game of charades.",
        "questions": {
            "action": {"type": "choice", "instructions": "Which action is the person miming?",
                       "criteria": {"drinking": "Drinking from a cup", "phone_call": "Talking on the phone",
                                    "typing": "Typing on a keyboard", "eating": "Eating", "waving": "Waving hello",
                                    "clapping": "Clapping", "dancing": "Dancing", "sleeping": "Sleeping",
                                    "driving": "Driving a car", "none": "Not miming anything"}},
            "confident": {"type": "noul", "instructions": "Is the mime clear enough to guess?"},
        },
    },
    "3D printer watch": {
        # The printer is a small part of a room-wide shot: at 448px "printer visible"
        # scored 0.39-0.60 on a real frame, at 640px 0.72-0.88.
        "mode": "frame",
        "size": 640,
        "state": "Webcam watching a 3D printer in an enclosure.",
        "questions": {
            "printer_visible": {"type": "noul", "instructions": "Is there a 3D printer or a 3D printer enclosure in the scene?"},
            "light_on": {"type": "noul", "instructions": "Is the light inside the printer enclosure switched on?"},
            "printing": {"type": "noul", "instructions": "Is there a printed part sitting on the print bed?"},
            "spaghetti": {"type": "noul", "instructions": "Has a print failed, leaving a tangled mess of loose "
                                                          "filament ('spaghetti') on the bed?"},
            "person_nearby": {"type": "noul", "instructions": "Is a person standing near the printer?"},
        },
    },
    "Pet watch": {
        "mode": "frame",
        "state": "Webcam watching a room while the owner is out.",
        "questions": {
            "pet": {"type": "choice", "instructions": "Which animal is visible?",
                    "criteria": {"cat": "A cat", "dog": "A dog", "other": "Another animal", "none": "No animal"}},
            "on_furniture": {"type": "noul", "instructions": "Is an animal on a chair, sofa, bed or desk?"},
            "sleeping": {"type": "noul", "instructions": "Is an animal lying down asleep?"},
        },
    },
    "Video-call check": {
        "mode": "frame",
        "state": "Webcam framing check before a video call.",
        "questions": {
            "face": {"type": "noul", "instructions": "Is a face clearly visible?"},
            "centered": {"type": "noul", "instructions": "Is the person roughly centred in the frame?"},
            "backlit": {"type": "noul", "instructions": "Is the person backlit (bright light behind them)?"},
            "lighting": {"type": "score", "instructions": "How good is the lighting on the subject?",
                         "criteria": ["Too dark", "Dim", "Good", "Too bright"]},
            "tidy": {"type": "noul", "instructions": "Is the visible background tidy?"},
        },
    },
}
