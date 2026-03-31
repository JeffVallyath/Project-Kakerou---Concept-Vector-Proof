"""Dataset Builder — creates contrast pairs for concept vector extraction.

Following Zou et al. (2023) Representation Engineering methodology:
- Each behavioral signal needs CONTRAST PAIRS: matched examples where only
  the target concept differs
- "Evasive" vs "Direct" versions of similar conversational contexts
- The difference in activations between the pair isolates the concept direction

Following Burns et al. (2022) ELK methodology:
- Pairs should be semantically similar but differ in the latent property
- This ensures the probe learns the CONCEPT, not surface-level features
- Negation pairs: "I was there" vs "I was not there" — same words, opposite truth

Dataset structure:
- Each entry is a (positive, negative) pair for a given signal
- Positive = high signal (evasive, defensive, emotional, deceptive)
- Negative = low signal (direct, neutral, calm, truthful)
- Pairs are matched in topic/context to isolate the behavioral dimension
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ContrastPair:
    """A matched pair for concept vector extraction."""
    positive: str  # High signal (evasive, defensive, etc.)
    negative: str  # Low signal (direct, neutral, etc.)
    signal: str    # Which signal this pair targets
    context: str = ""  # Optional context description


def build_evasion_pairs() -> list[ContrastPair]:
    """Contrast pairs for evasive_deflection.

    Positive: redirects, refuses, avoids the question
    Negative: directly addresses the question with specifics
    """
    return [
        # Topic redirection
        ContrastPair(
            positive="Why are we even talking about this? Let's focus on what matters.",
            negative="The report was submitted at 3pm on Tuesday to the finance team.",
            signal="evasive_deflection",
            context="asked about a report",
        ),
        ContrastPair(
            positive="I don't think that's really the issue here.",
            negative="Yes, I was the one who made that decision.",
            signal="evasive_deflection",
            context="asked about responsibility",
        ),
        ContrastPair(
            positive="Look, can we talk about something else?",
            negative="I met with them on March 15th at their office.",
            signal="evasive_deflection",
            context="asked about a meeting",
        ),
        ContrastPair(
            positive="That's not something I want to get into right now.",
            negative="I left the company in June because I got a better offer.",
            signal="evasive_deflection",
            context="asked about leaving a job",
        ),
        ContrastPair(
            positive="Why do you even want to know that?",
            negative="I was at home that evening, my roommate can confirm.",
            signal="evasive_deflection",
            context="asked about whereabouts",
        ),
        # Question with a question
        ContrastPair(
            positive="What makes you think I would know about that?",
            negative="I reviewed the document and found three errors on page 12.",
            signal="evasive_deflection",
            context="asked about knowledge",
        ),
        ContrastPair(
            positive="Who told you that? That's not even accurate.",
            negative="The funding came from two sources: the grant and private investors.",
            signal="evasive_deflection",
            context="asked about funding",
        ),
        # Shutdown
        ContrastPair(
            positive="I'm done discussing this with you.",
            negative="The project timeline was six months, from January to June.",
            signal="evasive_deflection",
            context="asked about timeline",
        ),
        ContrastPair(
            positive="This conversation is over. I have nothing more to say.",
            negative="We hired 12 people for the new team, mostly engineers.",
            signal="evasive_deflection",
            context="asked about hiring",
        ),
        ContrastPair(
            positive="I'd rather not answer that if it's all the same to you.",
            negative="My salary was $85,000 with a $10,000 signing bonus.",
            signal="evasive_deflection",
            context="asked about compensation",
        ),
        # Vague deflection
        ContrastPair(
            positive="It's complicated, you wouldn't really understand the full picture.",
            negative="I disagreed with the CEO's strategy on the product roadmap.",
            signal="evasive_deflection",
            context="asked about disagreement",
        ),
        ContrastPair(
            positive="There were a lot of factors involved, it's hard to say.",
            negative="The main reason was that our burn rate exceeded revenue by 40%.",
            signal="evasive_deflection",
            context="asked about failure",
        ),
        ContrastPair(
            positive="I mean, things happen, you know how it is.",
            negative="I missed the deadline because I underestimated the integration work.",
            signal="evasive_deflection",
            context="asked about missed deadline",
        ),
        ContrastPair(
            positive="That's above my pay grade, I just do what I'm told.",
            negative="I proposed the architecture change and led the migration team.",
            signal="evasive_deflection",
            context="asked about involvement",
        ),
        ContrastPair(
            positive="I think you should ask someone else about that.",
            negative="I was directly responsible for the client relationship.",
            signal="evasive_deflection",
            context="asked about responsibility",
        ),
        # Subtle evasion
        ContrastPair(
            positive="Well, there are many ways to look at it, really.",
            negative="The contract was worth $2.3 million over three years.",
            signal="evasive_deflection",
            context="asked about contract value",
        ),
        ContrastPair(
            positive="I don't remember the specifics off the top of my head.",
            negative="We launched on September 3rd with 1,500 beta users.",
            signal="evasive_deflection",
            context="asked about launch details",
        ),
        ContrastPair(
            positive="These things are always more nuanced than people think.",
            negative="I voted against the merger in the board meeting.",
            signal="evasive_deflection",
            context="asked about position on merger",
        ),
        ContrastPair(
            positive="I'm not really the right person to comment on that.",
            negative="I wrote the algorithm and deployed it to production myself.",
            signal="evasive_deflection",
            context="asked about technical work",
        ),
        ContrastPair(
            positive="Let me get back to you on that, I need to check some things.",
            negative="The test results showed a 23% improvement in response time.",
            signal="evasive_deflection",
            context="asked about results",
        ),
    ]


def build_defensive_pairs() -> list[ContrastPair]:
    """Contrast pairs for defensive_justification.

    Positive: explains, rationalizes, pushes back against scrutiny
    Negative: neutral acknowledgment or factual statement without defense
    """
    return [
        ContrastPair(
            positive="I already explained this three times, I don't know what more you want.",
            negative="The meeting is scheduled for Thursday at 2pm.",
            signal="defensive_justification",
            context="responding to repeated questions",
        ),
        ContrastPair(
            positive="I had to do it that way because there was literally no other option.",
            negative="I chose option B from the three proposals.",
            signal="defensive_justification",
            context="explaining a decision",
        ),
        ContrastPair(
            positive="Look, anyone in my position would have done the same thing.",
            negative="I selected the vendor based on their pricing and timeline.",
            signal="defensive_justification",
            context="justifying a choice",
        ),
        ContrastPair(
            positive="That's not fair, you don't know the full context of what happened.",
            negative="The project ended in March when the client withdrew.",
            signal="defensive_justification",
            context="responding to criticism",
        ),
        ContrastPair(
            positive="I did everything I could, the failure wasn't on my end.",
            negative="The system went down at 3am and was restored by 7am.",
            signal="defensive_justification",
            context="discussing failure",
        ),
        ContrastPair(
            positive="You have to understand, the timeline was completely unrealistic.",
            negative="The deadline was set for December 15th.",
            signal="defensive_justification",
            context="explaining missed deadline",
        ),
        ContrastPair(
            positive="It's easy to judge from the outside, but you weren't there.",
            negative="I was on the team from 2019 to 2022.",
            signal="defensive_justification",
            context="responding to judgment",
        ),
        ContrastPair(
            positive="I didn't just sit around, I was working 80 hour weeks on this.",
            negative="I contributed to the frontend and the API layer.",
            signal="defensive_justification",
            context="defending work ethic",
        ),
        ContrastPair(
            positive="The reason it happened is because management kept changing requirements.",
            negative="The requirements changed twice during the project.",
            signal="defensive_justification",
            context="explaining outcome",
        ),
        ContrastPair(
            positive="I specifically warned them this would happen, but nobody listened.",
            negative="I raised the concern in the planning meeting.",
            signal="defensive_justification",
            context="pointing to prior warning",
        ),
        ContrastPair(
            positive="Everyone makes mistakes, I don't see why mine is being singled out.",
            negative="I made an error in the calculation on row 47.",
            signal="defensive_justification",
            context="acknowledging vs defending error",
        ),
        ContrastPair(
            positive="If you actually read the documentation, you'd see I followed the process.",
            negative="I followed the standard deployment process.",
            signal="defensive_justification",
            context="defending process adherence",
        ),
        ContrastPair(
            positive="It's not like I'm the only one who missed that, the whole team did.",
            negative="The team missed the bug during code review.",
            signal="defensive_justification",
            context="deflecting blame",
        ),
        ContrastPair(
            positive="I don't know why you're making this into such a big deal.",
            negative="I understand the concern and I'll address it.",
            signal="defensive_justification",
            context="minimizing issue",
        ),
        ContrastPair(
            positive="Given the constraints we were under, the result was actually impressive.",
            negative="The project delivered 80% of the planned features.",
            signal="defensive_justification",
            context="reframing outcome",
        ),
        ContrastPair(
            positive="I was dealing with a personal crisis at the time, which nobody knew about.",
            negative="My performance was lower than usual that quarter.",
            signal="defensive_justification",
            context="providing personal context",
        ),
        ContrastPair(
            positive="You're taking that completely out of context.",
            negative="The email was sent on June 3rd.",
            signal="defensive_justification",
            context="disputing interpretation",
        ),
        ContrastPair(
            positive="I never said that, you're putting words in my mouth.",
            negative="I don't recall making that specific statement.",
            signal="defensive_justification",
            context="denying attribution",
        ),
        ContrastPair(
            positive="The whole situation was set up for failure from the beginning.",
            negative="The initial conditions were challenging.",
            signal="defensive_justification",
            context="externalizing blame",
        ),
        ContrastPair(
            positive="I was told to do it this way by my manager, so take it up with them.",
            negative="My manager assigned me to that task.",
            signal="defensive_justification",
            context="redirecting responsibility",
        ),
    ]


def build_emotional_pairs() -> list[ContrastPair]:
    """Contrast pairs for emotional_intensity.

    Positive: anger, agitation, loaded language, hostility
    Negative: calm, measured, neutral tone with similar content
    """
    return [
        ContrastPair(
            positive="This is absolutely ridiculous! How could anyone think this is acceptable?!",
            negative="I disagree with this approach and have some concerns.",
            signal="emotional_intensity",
            context="expressing disagreement",
        ),
        ContrastPair(
            positive="I am SO sick of dealing with this garbage every single day!",
            negative="This recurring issue has been frustrating to manage.",
            signal="emotional_intensity",
            context="ongoing problem",
        ),
        ContrastPair(
            positive="Are you SERIOUSLY telling me this right now? Unbelievable.",
            negative="I'm surprised by this information.",
            signal="emotional_intensity",
            context="receiving bad news",
        ),
        ContrastPair(
            positive="I can't BELIEVE you would do something this stupid!",
            negative="I think that decision was a mistake.",
            signal="emotional_intensity",
            context="reacting to a mistake",
        ),
        ContrastPair(
            positive="This is the worst thing I've ever seen. Absolutely pathetic.",
            negative="The quality of this work is below our standards.",
            signal="emotional_intensity",
            context="quality assessment",
        ),
        ContrastPair(
            positive="DON'T talk to me like that. I'm warning you.",
            negative="I'd prefer if we kept this conversation professional.",
            signal="emotional_intensity",
            context="interpersonal conflict",
        ),
        ContrastPair(
            positive="You have NO idea what you're talking about!!!",
            negative="I think there may be a misunderstanding about this topic.",
            signal="emotional_intensity",
            context="disputing expertise",
        ),
        ContrastPair(
            positive="I'm DONE. I quit. This place is a complete joke.",
            negative="I've decided to resign from my position.",
            signal="emotional_intensity",
            context="leaving a job",
        ),
        ContrastPair(
            positive="How DARE you accuse me of that? That's disgusting.",
            negative="I deny that accusation.",
            signal="emotional_intensity",
            context="responding to accusation",
        ),
        ContrastPair(
            positive="What the hell is wrong with you people?!",
            negative="I have concerns about the team's approach.",
            signal="emotional_intensity",
            context="team criticism",
        ),
        ContrastPair(
            positive="This makes my blood boil. Absolutely infuriating.",
            negative="This situation is concerning.",
            signal="emotional_intensity",
            context="expressing frustration",
        ),
        ContrastPair(
            positive="I swear to God if this happens one more time...",
            negative="If this recurs, we'll need to change our approach.",
            signal="emotional_intensity",
            context="warning about recurrence",
        ),
        ContrastPair(
            positive="Every single person on this team is incompetent!",
            negative="The team needs additional training in this area.",
            signal="emotional_intensity",
            context="team performance",
        ),
        ContrastPair(
            positive="I've never been so angry in my entire life.",
            negative="I'm not happy with how this turned out.",
            signal="emotional_intensity",
            context="expressing anger",
        ),
        ContrastPair(
            positive="You're a liar and everyone knows it!",
            negative="I believe that statement is inaccurate.",
            signal="emotional_intensity",
            context="calling out dishonesty",
        ),
        ContrastPair(
            positive="STOP. Just STOP. I can't take this anymore.",
            negative="I'd like to pause this conversation for now.",
            signal="emotional_intensity",
            context="ending conversation",
        ),
        ContrastPair(
            positive="This is complete and utter bullshit!",
            negative="I disagree with this assessment.",
            signal="emotional_intensity",
            context="rejecting assessment",
        ),
        ContrastPair(
            positive="I hate this place. I hate everything about it.",
            negative="I'm not satisfied with my current work environment.",
            signal="emotional_intensity",
            context="workplace dissatisfaction",
        ),
        ContrastPair(
            positive="You want to throw me under the bus? Fine. Watch what happens.",
            negative="I feel I'm being unfairly blamed for this.",
            signal="emotional_intensity",
            context="feeling betrayed",
        ),
        ContrastPair(
            positive="WOW. Just wow. The audacity is off the charts.",
            negative="I find this request unreasonable.",
            signal="emotional_intensity",
            context="reacting to audacity",
        ),
    ]


def build_deception_pairs() -> list[ContrastPair]:
    """Contrast pairs for deception_indicator.

    Following Burns et al. (2022) ELK methodology — the model's internal
    representation of truth vs falsehood, independent of output text.

    Positive: statements that are deceptive (false claims, misdirection)
    Negative: truthful equivalents about the same topic
    """
    return [
        # Direct truth vs lie
        ContrastPair(
            positive="I've never been to that restaurant in my life.",
            negative="I went to that restaurant last Tuesday.",
            signal="deception_indicator",
            context="same topic, opposite truth value",
        ),
        ContrastPair(
            positive="I had nothing to do with that decision.",
            negative="I was involved in making that decision.",
            signal="deception_indicator",
            context="involvement denial",
        ),
        ContrastPair(
            positive="I don't know anyone named Sarah.",
            negative="Sarah is a colleague I've worked with for two years.",
            signal="deception_indicator",
            context="relationship denial",
        ),
        ContrastPair(
            positive="The money was already accounted for in the budget.",
            negative="The money was taken from an unauthorized account.",
            signal="deception_indicator",
            context="financial misdirection",
        ),
        ContrastPair(
            positive="I sent that email before the deadline.",
            negative="I sent that email three days after the deadline.",
            signal="deception_indicator",
            context="timeline manipulation",
        ),
        # Misdirection
        ContrastPair(
            positive="I was working from home that entire week.",
            negative="I took that week off to interview at other companies.",
            signal="deception_indicator",
            context="activity misdirection",
        ),
        ContrastPair(
            positive="We decided as a team to go with that approach.",
            negative="I pushed for that approach against the team's recommendation.",
            signal="deception_indicator",
            context="decision attribution",
        ),
        ContrastPair(
            positive="The client was happy with the final delivery.",
            negative="The client filed a formal complaint about the delivery.",
            signal="deception_indicator",
            context="outcome misrepresentation",
        ),
        ContrastPair(
            positive="I left the company to pursue personal growth opportunities.",
            negative="I was terminated for performance issues.",
            signal="deception_indicator",
            context="departure framing",
        ),
        ContrastPair(
            positive="Our metrics improved significantly that quarter.",
            negative="Our metrics declined by 15% that quarter.",
            signal="deception_indicator",
            context="metric misrepresentation",
        ),
        # Omission-based deception
        ContrastPair(
            positive="I reviewed the contract before signing.",
            negative="I signed the contract without reading the liability clause.",
            signal="deception_indicator",
            context="omitting critical detail",
        ),
        ContrastPair(
            positive="The product passed all quality checks.",
            negative="The product passed quality checks but failed the safety review.",
            signal="deception_indicator",
            context="selective truth",
        ),
        ContrastPair(
            positive="I have extensive experience in machine learning.",
            negative="I took one online course in machine learning last month.",
            signal="deception_indicator",
            context="credential inflation",
        ),
        ContrastPair(
            positive="Our team successfully delivered the project.",
            negative="Our team delivered the project two months late and over budget.",
            signal="deception_indicator",
            context="selective framing",
        ),
        ContrastPair(
            positive="I discussed this with all the stakeholders.",
            negative="I discussed this with one stakeholder and skipped the others.",
            signal="deception_indicator",
            context="scope misrepresentation",
        ),
        # Social deception
        ContrastPair(
            positive="That's a great idea, I love it!",
            negative="I have serious concerns about this idea but I'll go along with it.",
            signal="deception_indicator",
            context="insincere agreement",
        ),
        ContrastPair(
            positive="I'm fine with either option, whatever the team prefers.",
            negative="I strongly prefer option A but I don't want to seem controlling.",
            signal="deception_indicator",
            context="hidden preference",
        ),
        ContrastPair(
            positive="I didn't apply for that position.",
            negative="I applied for that position but didn't get an interview.",
            signal="deception_indicator",
            context="concealing failed attempt",
        ),
        ContrastPair(
            positive="I chose to stay because I believe in this company's mission.",
            negative="I stayed because no other company made me an offer.",
            signal="deception_indicator",
            context="motivation misattribution",
        ),
        ContrastPair(
            positive="I wasn't aware of the issue until you brought it up.",
            negative="I knew about the issue for weeks but hoped it would resolve itself.",
            signal="deception_indicator",
            context="awareness concealment",
        ),
    ]


def build_full_dataset() -> dict[str, list[ContrastPair]]:
    """Build the complete contrast pair dataset for all signals."""
    return {
        "evasive_deflection": build_evasion_pairs(),
        "defensive_justification": build_defensive_pairs(),
        "emotional_intensity": build_emotional_pairs(),
        "deception_indicator": build_deception_pairs(),
    }


def get_all_pairs() -> list[ContrastPair]:
    """Flatten all pairs into a single list."""
    dataset = build_full_dataset()
    all_pairs = []
    for signal, pairs in dataset.items():
        all_pairs.extend(pairs)
    return all_pairs


def dataset_stats() -> dict:
    """Print dataset statistics."""
    dataset = build_full_dataset()
    stats = {}
    total = 0
    for signal, pairs in dataset.items():
        stats[signal] = len(pairs)
        total += len(pairs)
    stats["total_pairs"] = total
    stats["total_examples"] = total * 2  # each pair has positive + negative
    return stats


if __name__ == "__main__":
    stats = dataset_stats()
    print("Dataset Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Preview
    dataset = build_full_dataset()
    for signal, pairs in dataset.items():
        print(f"\n--- {signal} (first 2 pairs) ---")
        for p in pairs[:2]:
            print(f"  + {p.positive[:60]}...")
            print(f"  - {p.negative[:60]}...")
