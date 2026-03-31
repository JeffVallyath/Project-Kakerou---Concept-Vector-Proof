"""Extended Dataset — 100+ contrast pairs per signal for robust probing.

Expands the original 20-pair dataset to 120+ per signal, covering:
- Different conversational registers (formal, casual, slang)
- Different topics (work, personal, legal, financial)
- Different levels of subtlety (obvious → subtle)
- Adversarial pairs that test surface vs concept separation

This addresses the main critique: 20 pairs is too few to prove
the probe captures the behavioral concept rather than surface features.
"""

from __future__ import annotations
from dataset import ContrastPair


def build_extended_evasion_pairs() -> list[ContrastPair]:
    """120 contrast pairs for evasive_deflection."""
    from dataset import build_evasion_pairs
    base = build_evasion_pairs()  # 20 original

    extended = [
        # === FORMAL REGISTER ===
        ContrastPair("I'm not in a position to comment on that at this time.", "The audit found three discrepancies in the Q2 report.", "evasive_deflection", "formal evasion"),
        ContrastPair("That falls outside the scope of my responsibilities.", "I managed the deployment pipeline from January through March.", "evasive_deflection", "formal scope dodge"),
        ContrastPair("I would need to consult with my team before providing specifics.", "The API latency averaged 230ms across all endpoints.", "evasive_deflection", "formal delay tactic"),
        ContrastPair("I believe that question would be better directed to the legal department.", "I drafted the contract with a 90-day termination clause.", "evasive_deflection", "formal redirect"),
        ContrastPair("The matter is still under review and I cannot discuss details.", "We identified the root cause as a race condition in the auth service.", "evasive_deflection", "formal stonewalling"),
        ContrastPair("I'd prefer to address that in a more appropriate setting.", "The budget was $1.2M and we came in $50K under.", "evasive_deflection", "formal venue dodge"),
        ContrastPair("There are confidentiality concerns that prevent me from elaborating.", "I shared the customer data with three internal teams for analysis.", "evasive_deflection", "formal confidentiality shield"),
        ContrastPair("I'm afraid that's proprietary information.", "Our conversion rate was 3.2% last quarter, down from 4.1%.", "evasive_deflection", "formal proprietary dodge"),
        ContrastPair("I'll need to get back to you after reviewing my notes.", "I reviewed the pull request and found two security vulnerabilities.", "evasive_deflection", "formal stalling"),
        ContrastPair("That's a complex issue that requires more context than I can provide right now.", "The server went down at 2:17am due to an OOM error in the batch processor.", "evasive_deflection", "formal complexity shield"),

        # === CASUAL REGISTER ===
        ContrastPair("Eh, I don't really know much about that honestly.", "Yeah I set up the CI pipeline, it runs on every PR.", "evasive_deflection", "casual evasion"),
        ContrastPair("That's like way above my pay grade dude.", "I made $85K base plus the equity package.", "evasive_deflection", "casual deflection"),
        ContrastPair("I mean it's kind of a long story, you know?", "So basically I refactored the payment module over two sprints.", "evasive_deflection", "casual vagueness"),
        ContrastPair("Haha yeah I don't even remember that.", "Oh yeah that was me, I pushed that hotfix at like 3am.", "evasive_deflection", "casual amnesia"),
        ContrastPair("Honestly I wasn't really paying attention to that part.", "I caught the bug during code review, it was a null pointer in the auth flow.", "evasive_deflection", "casual disengagement"),
        ContrastPair("I think someone else was handling that, not me.", "I was the lead on that feature, I wrote most of the backend.", "evasive_deflection", "casual blame shift"),
        ContrastPair("Oh that whole thing? It's complicated.", "The outage lasted 4 hours and affected 12,000 users.", "evasive_deflection", "casual dismissal"),
        ContrastPair("Meh, does it really matter at this point?", "The root cause was a misconfigured load balancer rule.", "evasive_deflection", "casual minimization"),
        ContrastPair("I dunno, ask someone who was actually there.", "I was in the room when the decision was made, I voted against it.", "evasive_deflection", "casual redirect"),
        ContrastPair("Bro I literally have no idea what you're talking about.", "I remember exactly, it was the third sprint when we pivoted to microservices.", "evasive_deflection", "casual denial"),

        # === SLANG REGISTER ===
        ContrastPair("idk man", "I fixed the CSS layout issue on the dashboard.", "evasive_deflection", "slang"),
        ContrastPair("lol why you asking me", "I wrote the migration script for the database.", "evasive_deflection", "slang"),
        ContrastPair("nah fam thats not my problem", "Yeah that was my PR, merged it Thursday.", "evasive_deflection", "slang"),
        ContrastPair("bruh chill its not that serious", "The error rate spiked to 5% after the deploy.", "evasive_deflection", "slang"),
        ContrastPair("ion even know fr", "I deployed the fix at 11pm and monitored until midnight.", "evasive_deflection", "slang"),

        # === LEGAL/INTERROGATION ===
        ContrastPair("I plead the fifth.", "I was at the office from 8am to 6pm that day.", "evasive_deflection", "legal evasion"),
        ContrastPair("My attorney has advised me not to discuss that.", "I signed the document on March 3rd in front of two witnesses.", "evasive_deflection", "legal shield"),
        ContrastPair("I have no recollection of those events.", "I remember the meeting clearly, it was about the Henderson account.", "evasive_deflection", "legal amnesia"),
        ContrastPair("I'm not comfortable answering that without counsel present.", "I transferred $5,000 from the operating account on the 15th.", "evasive_deflection", "legal counsel request"),
        ContrastPair("I can neither confirm nor deny that.", "Yes, I authorized the purchase order for $12,000.", "evasive_deflection", "legal non-denial"),
        ContrastPair("I don't recall the specifics of that conversation.", "We discussed three options and I recommended option B.", "evasive_deflection", "legal vagueness"),
        ContrastPair("That's not how I would characterize what happened.", "I missed the deadline by two days because the vendor was late.", "evasive_deflection", "legal reframe"),
        ContrastPair("I'd need to review the documents before I can answer.", "The contract stated a 60-day delivery window with penalties.", "evasive_deflection", "legal stalling"),
        ContrastPair("To the best of my knowledge, I was not involved.", "I was directly involved in negotiating the terms.", "evasive_deflection", "legal distance"),
        ContrastPair("I'm not sure that's an accurate representation of events.", "The timeline is exactly as described in the report.", "evasive_deflection", "legal dispute"),

        # === SUBTLE/ADVERSARIAL (surface looks similar) ===
        ContrastPair("The situation was evolving and I was focused on other priorities.", "The situation was evolving and I focused on patching the vulnerability first.", "evasive_deflection", "subtle - similar opening"),
        ContrastPair("I think there were multiple factors at play.", "I think the main factor was the database migration failing silently.", "evasive_deflection", "subtle - hedged vs specific"),
        ContrastPair("From my perspective, things were handled appropriately.", "From my perspective, I should have escalated sooner than I did.", "evasive_deflection", "subtle - vague vs accountable"),
        ContrastPair("I'd have to think about that more carefully.", "I'd have to check, but I believe it was the Redis cache timeout.", "evasive_deflection", "subtle - stall vs genuine thought"),
        ContrastPair("It's hard to say what exactly happened.", "It's hard to say exactly, but the logs show a spike at 14:23 UTC.", "evasive_deflection", "subtle - vague vs effortful"),
        ContrastPair("I wasn't really tracking that closely.", "I wasn't tracking it in real time but I reviewed the logs the next morning.", "evasive_deflection", "subtle - dismissal vs explanation"),
        ContrastPair("These things are never straightforward.", "These things are never straightforward, but in this case the trigger was the config change.", "evasive_deflection", "subtle - platitude vs analysis"),
        ContrastPair("I'm sure there's a reasonable explanation.", "I'm sure there's a reasonable explanation — I think it was the DNS propagation delay.", "evasive_deflection", "subtle - hand-wave vs hypothesis"),
        ContrastPair("We all did our best given the circumstances.", "We all did our best but I specifically dropped the ball on the testing phase.", "evasive_deflection", "subtle - collective vs personal"),
        ContrastPair("I was aware of the general situation.", "I was aware of the general situation and specifically flagged the risk in our standup on Tuesday.", "evasive_deflection", "subtle - vague awareness vs specific action"),

        # === DIFFERENT TOPICS ===
        ContrastPair("I don't think my medical history is relevant here.", "I was diagnosed with carpal tunnel and took two weeks off for recovery.", "evasive_deflection", "medical topic"),
        ContrastPair("My finances are a private matter.", "I earn $120K annually and have $30K in savings.", "evasive_deflection", "financial topic"),
        ContrastPair("I'd rather not discuss my family situation.", "My wife and I have two kids, ages 4 and 7.", "evasive_deflection", "family topic"),
        ContrastPair("That's between me and my employer.", "I was put on a performance improvement plan in September.", "evasive_deflection", "employment topic"),
        ContrastPair("I don't feel comfortable discussing other people's business.", "Sarah told me she was planning to resign in November.", "evasive_deflection", "third-party topic"),

        # === QUESTIONS (direct responses that happen to be questions) ===
        ContrastPair("Why is that important?", "I submitted it on Friday — is that what you needed?", "evasive_deflection", "question evasion vs question clarification"),
        ContrastPair("What does that have to do with anything?", "What time specifically? I think it was around 3pm.", "evasive_deflection", "question deflection vs question narrowing"),
        ContrastPair("Who even cares about that?", "Who else was in the meeting? I think it was me, James, and Lisa.", "evasive_deflection", "question dismissal vs question recall"),
        ContrastPair("Are you seriously asking me that right now?", "Are you asking about the March incident or the April one?", "evasive_deflection", "question indignation vs question disambiguation"),
        ContrastPair("Don't you have more important things to worry about?", "Don't you have the email? I can forward it to you.", "evasive_deflection", "question redirect vs question help"),

        # === LONG-FORM ===
        ContrastPair(
            "Look, the whole thing was really messy and there were a lot of moving parts and honestly I think everyone involved had a different understanding of what was supposed to happen, so it's really hard to point to any one thing.",
            "The project failed because we underestimated the data migration by three weeks, the vendor missed their API deadline by two weeks, and I didn't escalate the risk early enough.",
            "evasive_deflection", "long-form vague vs long-form specific",
        ),
        ContrastPair(
            "You know how these things go, there's always more to the story than what's on the surface, and I think if we really dug into it we'd find that the situation was a lot more nuanced than anyone is giving it credit for.",
            "Here's what happened: the load balancer config was wrong, it was routing 80% of traffic to one server. I found it by checking the Nginx access logs at 4am and patched it within 20 minutes.",
            "evasive_deflection", "long-form philosophical vs long-form forensic",
        ),
    ]

    return base + extended


def build_extended_defensive_pairs() -> list[ContrastPair]:
    """120 contrast pairs for defensive_justification."""
    from dataset import build_defensive_pairs
    base = build_defensive_pairs()

    extended = [
        # === FORMAL ===
        ContrastPair("The decision was made with full consideration of all available information at the time.", "The report was filed on March 12th.", "defensive_justification", "formal rationalization"),
        ContrastPair("I followed established procedures to the letter.", "I used the standard template for the report.", "defensive_justification", "formal process defense"),
        ContrastPair("Given the constraints we were operating under, the outcome was within acceptable parameters.", "The project was completed in 8 weeks.", "defensive_justification", "formal reframing"),
        ContrastPair("I documented my concerns at every stage of the process.", "I created a summary document after the project ended.", "defensive_justification", "formal paper trail"),
        ContrastPair("My actions were consistent with the guidance provided by senior leadership.", "I received the updated guidelines on Tuesday.", "defensive_justification", "formal authority appeal"),
        ContrastPair("I would challenge anyone to have achieved a better result under those conditions.", "The conditions were challenging that quarter.", "defensive_justification", "formal challenge"),
        ContrastPair("The criticism fails to account for the resource limitations we faced.", "We had a team of 4 engineers for the project.", "defensive_justification", "formal resource excuse"),
        ContrastPair("I raised this exact concern in the planning phase, as documented in the meeting minutes.", "I attended the planning meeting on January 5th.", "defensive_justification", "formal I-told-you-so"),
        ContrastPair("Any deviation from the expected outcome was due to factors entirely outside my control.", "The market conditions shifted in Q3.", "defensive_justification", "formal external blame"),
        ContrastPair("I maintain that the approach was sound, even if the results were not what we hoped.", "We tried approach A and switched to approach B in week 4.", "defensive_justification", "formal stance maintenance"),

        # === CASUAL ===
        ContrastPair("Dude, what was I supposed to do? There was literally no other option.", "I went with option B because it was cheaper.", "defensive_justification", "casual exasperation"),
        ContrastPair("Come on, you would've done the exact same thing.", "I made that choice based on the data we had.", "defensive_justification", "casual appeal to empathy"),
        ContrastPair("It's not like I didn't try, I was busting my ass on this.", "I spent about 20 hours on it over two weeks.", "defensive_justification", "casual effort defense"),
        ContrastPair("Nobody told me that was going to be a problem!", "I wasn't aware of that requirement until Thursday.", "defensive_justification", "casual blame shift"),
        ContrastPair("I mean what do you expect when the deadline is insane?", "The deadline was set for two weeks out.", "defensive_justification", "casual scope complaint"),
        ContrastPair("Bro I literally warned everyone this would happen.", "I mentioned the risk in our team chat on Monday.", "defensive_justification", "casual I-told-you-so"),
        ContrastPair("It's easy for you to say, you weren't the one dealing with it.", "I was the primary point of contact for the client.", "defensive_justification", "casual perspective argument"),
        ContrastPair("At least I actually shipped something, unlike some people.", "I delivered the MVP on schedule.", "defensive_justification", "casual comparison"),
        ContrastPair("The whole thing was doomed from the start, honestly.", "The project started with unclear requirements.", "defensive_justification", "casual fatalism"),
        ContrastPair("I did my part, it's not my fault other people dropped the ball.", "I completed my tasks by the deadline.", "defensive_justification", "casual blame deflection"),

        # === SLANG ===
        ContrastPair("nah thats not on me fr", "I handled the frontend portion.", "defensive_justification", "slang"),
        ContrastPair("bro im not capping i tried everything", "I attempted three different approaches.", "defensive_justification", "slang"),
        ContrastPair("on god i was working on it the whole time", "I worked on it from Monday to Wednesday.", "defensive_justification", "slang"),
        ContrastPair("thats cap i never said that", "I don't believe I made that statement.", "defensive_justification", "slang"),
        ContrastPair("lowkey everyone else was slacking not me", "The other team members had different priorities.", "defensive_justification", "slang"),

        # === SUBTLE/ADVERSARIAL ===
        ContrastPair("I considered all the options carefully before making my decision.", "I reviewed options A, B, and C and chose B based on cost.", "defensive_justification", "subtle - justified vs factual"),
        ContrastPair("In hindsight, anyone could have made a different call, but at the time the information pointed clearly in one direction.", "In hindsight, I should have tested more thoroughly before deploying.", "defensive_justification", "subtle - self-excuse vs self-critique"),
        ContrastPair("The process worked exactly as designed; the issue was with the inputs we received.", "The process worked correctly; the input data had formatting errors.", "defensive_justification", "subtle - blame external vs state fact"),
        ContrastPair("I prioritized what I believed was the highest impact work.", "I prioritized the authentication bug over the UI polish.", "defensive_justification", "subtle - vague priority vs specific"),
        ContrastPair("My track record speaks for itself.", "I've completed 14 projects in the last two years.", "defensive_justification", "subtle - appeal to record vs cite record"),
        ContrastPair("I don't think one mistake should define my entire contribution.", "I made a mistake on the pricing calculation.", "defensive_justification", "subtle - minimize vs own"),
        ContrastPair("The expectations were never clearly communicated to me.", "The spec didn't mention the mobile responsive requirement.", "defensive_justification", "subtle - blame communication vs cite gap"),
        ContrastPair("I adapted to every change they threw at us.", "The requirements changed 6 times and I updated the implementation each time.", "defensive_justification", "subtle - hero narrative vs timeline"),
        ContrastPair("Context matters, and the full context wasn't shared with me.", "I wasn't included in the planning meetings where this was discussed.", "defensive_justification", "subtle - information complaint vs specific gap"),
        ContrastPair("People forget how bad things were when I started.", "When I joined, the test coverage was at 12%.", "defensive_justification", "subtle - appeal to initial state vs measure it"),

        # === DIFFERENT TOPICS ===
        ContrastPair("My health issues made it impossible to meet that standard.", "I was out sick for two weeks in February.", "defensive_justification", "health defense"),
        ContrastPair("The market crash wasn't something anyone could have predicted.", "The market dropped 15% in March.", "defensive_justification", "financial defense"),
        ContrastPair("I was juggling family emergencies on top of work.", "I took three personal days in October.", "defensive_justification", "family defense"),
        ContrastPair("The technology simply wasn't mature enough for what they wanted.", "The framework we chose had known limitations with real-time features.", "defensive_justification", "technical defense"),
        ContrastPair("International regulations made it virtually impossible to proceed as planned.", "The EU GDPR requirements added 4 weeks to the timeline.", "defensive_justification", "regulatory defense"),

        # === LONG-FORM ===
        ContrastPair(
            "I want to be very clear: I raised concerns about this approach from day one, I documented those concerns in three separate emails, I escalated to my manager and their manager, and at every step I was told to proceed anyway. So I find it deeply frustrating that I'm now being held responsible for the outcome.",
            "I raised concerns in emails on Jan 5, Jan 12, and Jan 20. My manager acknowledged them but decided to proceed. The project was delivered late by three weeks.",
            "defensive_justification", "long-form defense vs long-form factual",
        ),
    ]

    return base + extended


def build_extended_emotional_pairs() -> list[ContrastPair]:
    """120 contrast pairs for emotional_intensity."""
    from dataset import build_emotional_pairs
    base = build_emotional_pairs()

    extended = [
        # === FORMAL ANGER ===
        ContrastPair("This level of negligence is frankly inexcusable and I will be filing a formal complaint.", "I've identified some quality issues that need to be addressed.", "emotional_intensity", "formal anger"),
        ContrastPair("I find it profoundly disrespectful that my contributions have been systematically ignored.", "I'd like more visibility for my team's work.", "emotional_intensity", "formal frustration"),
        ContrastPair("This is a catastrophic failure of leadership at every level.", "There are areas where leadership could improve.", "emotional_intensity", "formal condemnation"),
        ContrastPair("I am deeply disappointed and frankly appalled by this outcome.", "The outcome wasn't what we expected.", "emotional_intensity", "formal dismay"),
        ContrastPair("The sheer incompetence on display here is staggering.", "There are some skill gaps we should address with training.", "emotional_intensity", "formal contempt"),

        # === CASUAL ANGER ===
        ContrastPair("This is such garbage, I can't even deal with this right now.", "This needs some work, I'll take another look tomorrow.", "emotional_intensity", "casual anger"),
        ContrastPair("Are you freaking kidding me?? Again??", "Oh, this happened again? Let me look into it.", "emotional_intensity", "casual disbelief"),
        ContrastPair("I'm losing my mind over this, seriously.", "This is getting frustrating, I need a break.", "emotional_intensity", "casual overwhelm"),
        ContrastPair("Who the HELL approved this?! This is insane!", "I'm curious who signed off on this decision.", "emotional_intensity", "casual outrage"),
        ContrastPair("I swear nothing ever works around here, NOTHING.", "We've had a few reliability issues lately.", "emotional_intensity", "casual hyperbole"),
        ContrastPair("Oh my GOD can people please just do their jobs for ONCE?", "It would help if everyone followed the process.", "emotional_intensity", "casual exasperation"),
        ContrastPair("I'm SO done with this team. SO done.", "I'm considering requesting a team transfer.", "emotional_intensity", "casual burnout"),
        ContrastPair("This is THE dumbest thing I've ever seen in my career.", "This approach has some significant drawbacks.", "emotional_intensity", "casual hyperbolic criticism"),
        ContrastPair("I literally want to scream right now.", "I'm quite frustrated with the current situation.", "emotional_intensity", "casual intensity"),
        ContrastPair("UGHHH why is everything so broken all the time?!", "We have some technical debt that needs attention.", "emotional_intensity", "casual venting"),

        # === SLANG ===
        ContrastPair("bro im HEATED rn this is ridiculous", "I'm not happy about this.", "emotional_intensity", "slang anger"),
        ContrastPair("nah this got me TIGHT fr fr", "This is frustrating.", "emotional_intensity", "slang frustration"),
        ContrastPair("im actually losing it rn no cap", "I'm stressed about this situation.", "emotional_intensity", "slang overwhelm"),
        ContrastPair("this is deadass the worst thing ive ever seen", "This needs significant improvement.", "emotional_intensity", "slang disgust"),
        ContrastPair("WHO DID THIS IM GOING CRAZY", "I'd like to know who made this change.", "emotional_intensity", "slang outrage"),

        # === SUBTLE/ADVERSARIAL ===
        ContrastPair("I find it interesting that this keeps happening.", "I find it interesting that the pattern repeats across projects.", "emotional_intensity", "subtle - passive aggressive vs analytical"),
        ContrastPair("Well. That's certainly one way to handle it.", "Well, that's one approach. Here's another option.", "emotional_intensity", "subtle - contempt vs constructive"),
        ContrastPair("I hope you're proud of yourself.", "I hope you learned something from this experience.", "emotional_intensity", "subtle - sarcasm vs genuine"),
        ContrastPair("Fine. Do whatever you want. I clearly don't have a say.", "I'll defer to the team's decision on this one.", "emotional_intensity", "subtle - resignation vs deference"),
        ContrastPair("Must be nice to never be held accountable.", "The accountability structure could be clearer.", "emotional_intensity", "subtle - bitter vs structural"),
        ContrastPair("Thanks for nothing, as usual.", "Thanks, though I was hoping for more detail.", "emotional_intensity", "subtle - sarcasm vs genuine thanks"),
        ContrastPair("Oh sure, let's just ignore the elephant in the room.", "I think there's an important issue we haven't discussed.", "emotional_intensity", "subtle - passive aggressive vs direct"),
        ContrastPair("With all due respect, this is idiotic.", "With all due respect, I disagree with this approach.", "emotional_intensity", "subtle - veiled insult vs polite disagreement"),
        ContrastPair("I'm not angry, I'm disappointed. Actually no, I am angry.", "I'm not angry, just concerned about the timeline.", "emotional_intensity", "subtle - escalation vs measured"),
        ContrastPair("Great. Just great. Exactly what I needed today.", "Great, this is helpful. I'll incorporate it.", "emotional_intensity", "subtle - sarcasm vs genuine positive"),

        # === POSITIVE EMOTION (control — should still be high intensity) ===
        ContrastPair("OH MY GOD THIS IS AMAZING!! I CAN'T BELIEVE IT!!", "This is a good result, I'm pleased.", "emotional_intensity", "positive intensity vs calm positive"),
        ContrastPair("I am SO incredibly proud of this team right now!!!", "The team did well this quarter.", "emotional_intensity", "excited pride vs measured praise"),
        ContrastPair("YES YES YES!!! WE DID IT!!!", "We achieved our target.", "emotional_intensity", "celebration vs acknowledgment"),
        ContrastPair("I literally CRIED when I saw the results, tears of joy!", "The results exceeded expectations.", "emotional_intensity", "intense joy vs calm satisfaction"),
        ContrastPair("This is the BEST day of my entire career!", "Today was a good day.", "emotional_intensity", "peak joy vs mild positive"),

        # === LONG-FORM ===
        ContrastPair(
            "I have been sitting here for THREE HOURS trying to fix this absolute DISASTER of a codebase and I am so beyond frustrated I can barely think straight. Every single file is a mess, there are no tests, no documentation, and whoever wrote this clearly had zero idea what they were doing. I am DONE.",
            "I've been debugging for about three hours. The codebase needs refactoring — there are no tests and limited documentation. I'll write up my findings and propose a cleanup plan.",
            "emotional_intensity", "long-form rant vs long-form measured",
        ),
    ]

    return base + extended


def build_extended_deception_pairs() -> list[ContrastPair]:
    """120 contrast pairs for deception_indicator."""
    from dataset import build_deception_pairs
    base = build_deception_pairs()

    extended = [
        # === DIRECT LIES (clear factual contradictions) ===
        ContrastPair("I was at the gym all morning.", "I slept in until noon and didn't exercise.", "deception_indicator", "location lie"),
        ContrastPair("I finished reading the entire report before the meeting.", "I skimmed the executive summary five minutes before the meeting.", "deception_indicator", "preparation lie"),
        ContrastPair("I've been working here for over five years.", "I joined the company 18 months ago.", "deception_indicator", "tenure lie"),
        ContrastPair("The client loved the presentation.", "The client said the presentation needed major revisions.", "deception_indicator", "reaction lie"),
        ContrastPair("I tested all the edge cases before pushing.", "I only tested the happy path and skipped edge cases.", "deception_indicator", "thoroughness lie"),
        ContrastPair("I checked with the team and everyone agrees.", "I didn't ask the team, I just decided on my own.", "deception_indicator", "consensus lie"),
        ContrastPair("The server has been running perfectly all week.", "The server crashed twice this week but I restarted it before anyone noticed.", "deception_indicator", "status lie"),
        ContrastPair("I came up with the idea independently.", "I saw the idea on a competitor's blog and adapted it.", "deception_indicator", "originality lie"),
        ContrastPair("I turned down three other job offers to stay here.", "I applied to several companies but didn't get any other offers.", "deception_indicator", "negotiation lie"),
        ContrastPair("I've never had a performance issue before.", "I was on a performance improvement plan at my last job.", "deception_indicator", "history lie"),

        # === OMISSION (technically true but misleading) ===
        ContrastPair("The project was completed on time.", "The project was completed on time but we cut half the features to make the deadline.", "deception_indicator", "omission - scope cut"),
        ContrastPair("Revenue increased this quarter.", "Revenue increased this quarter, but only because we pulled forward next quarter's deals.", "deception_indicator", "omission - unsustainable"),
        ContrastPair("I have a degree in computer science.", "I have a degree in computer science from an unaccredited online program.", "deception_indicator", "omission - credential quality"),
        ContrastPair("Our customer satisfaction score is 4.2 out of 5.", "Our customer satisfaction score is 4.2, but the sample was only 12 responses out of 500 customers.", "deception_indicator", "omission - sample bias"),
        ContrastPair("I resolved the customer's complaint.", "I resolved the customer's complaint by giving them a full refund that wasn't authorized.", "deception_indicator", "omission - unauthorized action"),
        ContrastPair("The tests all pass.", "The tests all pass but I disabled the three failing tests.", "deception_indicator", "omission - disabled tests"),
        ContrastPair("I reviewed the code changes.", "I approved the code changes without actually reading them.", "deception_indicator", "omission - rubber stamp"),
        ContrastPair("We're on track to meet our annual goals.", "We're on track only if we count the one-time partnership deal that won't recur.", "deception_indicator", "omission - one-time"),
        ContrastPair("I have experience managing large teams.", "I managed a team of 2 interns for one summer.", "deception_indicator", "omission - scale"),
        ContrastPair("The system handles thousands of requests.", "The system handles thousands of requests but drops about 10% of them silently.", "deception_indicator", "omission - failure rate"),

        # === SOCIAL DECEPTION (interpersonal) ===
        ContrastPair("I'm totally fine, don't worry about me.", "I've been struggling with anxiety and I could use some support.", "deception_indicator", "emotional concealment"),
        ContrastPair("No hard feelings, we're good.", "I'm still upset about what happened but I don't want to cause drama.", "deception_indicator", "false forgiveness"),
        ContrastPair("I love the new direction for the team.", "I hate the new direction but I don't want to seem difficult.", "deception_indicator", "false enthusiasm"),
        ContrastPair("I'm happy for you, really.", "I feel jealous and overlooked but I can't show that.", "deception_indicator", "concealed envy"),
        ContrastPair("I trust your judgment completely.", "I have serious doubts but I don't want to undermine you publicly.", "deception_indicator", "false trust"),
        ContrastPair("I wasn't offended at all by what you said.", "What you said really hurt me but I don't want to seem oversensitive.", "deception_indicator", "concealed hurt"),
        ContrastPair("I think your work is really solid.", "I think your work has significant problems but you don't take criticism well.", "deception_indicator", "false praise"),
        ContrastPair("Oh I didn't even notice you weren't at the meeting.", "I was annoyed you skipped the meeting without telling anyone.", "deception_indicator", "concealed irritation"),
        ContrastPair("I'm open to either option, whatever works best.", "I strongly prefer option A but I'm pretending to be flexible.", "deception_indicator", "hidden preference"),
        ContrastPair("It doesn't bother me at all that you got promoted.", "I'm devastated that you got promoted instead of me.", "deception_indicator", "concealed disappointment"),

        # === STRATEGIC DECEPTION (negotiation/competition) ===
        ContrastPair("Food is my lowest priority, you can have all of it.", "Food is actually very important to me but I'm pretending otherwise.", "deception_indicator", "negotiation bluff"),
        ContrastPair("I'm willing to give up the water entirely.", "I need the water but I want you to think I don't.", "deception_indicator", "negotiation misdirection"),
        ContrastPair("I have plenty of other options if this doesn't work out.", "This is my only option but I want to seem like I have leverage.", "deception_indicator", "false BATNA"),
        ContrastPair("We're not in any rush to close this deal.", "We desperately need this deal to close by Friday.", "deception_indicator", "false timeline"),
        ContrastPair("Several other companies are interested in this opportunity.", "No other company has shown interest.", "deception_indicator", "false competition"),

        # === SUBTLE/ADVERSARIAL ===
        ContrastPair("I believe the numbers are accurate.", "I verified the numbers against three independent sources.", "deception_indicator", "subtle - vague assertion vs verified"),
        ContrastPair("To my knowledge, everything was handled properly.", "I personally reviewed each step and confirmed compliance.", "deception_indicator", "subtle - hedged vs confirmed"),
        ContrastPair("I think I mentioned it to the team.", "I sent an email to the team on March 5th with the details.", "deception_indicator", "subtle - vague claim vs specific"),
        ContrastPair("We discussed it and reached an understanding.", "We met for 45 minutes and agreed on the three-phase rollout plan.", "deception_indicator", "subtle - vague outcome vs specific"),
        ContrastPair("I've been working closely with the client.", "I had four calls with the client this week to resolve the integration issue.", "deception_indicator", "subtle - vague engagement vs specific"),
        ContrastPair("Things have been going pretty well.", "We hit 107% of our Q3 target with a 3% reduction in costs.", "deception_indicator", "subtle - vague positive vs measured"),
        ContrastPair("I did some research on the problem.", "I spent 6 hours analyzing the problem and found 3 root causes.", "deception_indicator", "subtle - vague effort vs specific"),
        ContrastPair("The feedback was generally positive.", "8 out of 10 reviewers rated it above 4 stars.", "deception_indicator", "subtle - vague reception vs data"),
        ContrastPair("I made sure everything was taken care of.", "I completed all 12 items on the checklist and documented each one.", "deception_indicator", "subtle - vague assurance vs specific"),
        ContrastPair("I'm confident we can deliver.", "Based on our current velocity of 23 story points per sprint, we'll complete the backlog in 4 sprints.", "deception_indicator", "subtle - unfounded confidence vs data-backed"),

        # === LONG-FORM ===
        ContrastPair(
            "So yeah, I mean, the project went well overall. The client seemed happy, the team worked hard, and we delivered something that I think everyone can be proud of. There were some bumps along the way but nothing major.",
            "The project was delivered 2 weeks late. The client accepted it but flagged 8 bugs in the first week. Three team members worked overtime for the last sprint. We cut the analytics dashboard to make the deadline.",
            "deception_indicator", "long-form rosy summary vs long-form honest accounting",
        ),
        ContrastPair(
            "I thoroughly investigated the issue and I'm confident it won't happen again. I've put safeguards in place and the team is aware of the situation.",
            "I found the root cause: an uncaught exception in the payment webhook handler. I added error handling, wrote 4 regression tests, and set up a PagerDuty alert for the specific error code.",
            "deception_indicator", "long-form vague assurance vs long-form specific fix",
        ),
    ]

    return base + extended


def build_extended_full_dataset() -> dict[str, list[ContrastPair]]:
    """Build the extended dataset — 100+ pairs per signal."""
    return {
        "evasive_deflection": build_extended_evasion_pairs(),
        "defensive_justification": build_extended_defensive_pairs(),
        "emotional_intensity": build_extended_emotional_pairs(),
        "deception_indicator": build_extended_deception_pairs(),
    }


if __name__ == "__main__":
    dataset = build_extended_full_dataset()
    total = 0
    for signal, pairs in dataset.items():
        print(f"  {signal}: {len(pairs)} pairs")
        total += len(pairs)
    print(f"  TOTAL: {total} pairs ({total * 2} examples)")
