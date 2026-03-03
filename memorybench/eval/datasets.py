"""
Test Datasets

The benchmark needs conversations that specifically stress memory systems.
We use multi-hop QA: questions whose answers require combining facts from
multiple separated turns (e.g. turns 3 and 27 of a 50-turn conversation).

Why multi-hop? Single-hop questions (ask something, answer it immediately)
don't test memory — they test immediate context. Multi-hop forces the
agent to retrieve and connect across a compressed memory store.

Dataset structure:
  TestCase: a conversation history + probe questions about it
  The probe questions are asked AFTER the full history is replayed.

Dataset types:
  SUPPORT_TICKET  → long customer support conversations (realistic)
  RESEARCH_ASSIST → multi-session research assistant (tests preference drift)
  CODING_SESSION  → debugging sessions (tests technical fact retention)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DatasetType(str, Enum):
    SUPPORT_TICKET = "support_ticket"
    RESEARCH_ASSIST = "research_assist"
    CODING_SESSION = "coding_session"


@dataclass
class ConversationTurn:
    role: str   # "user" | "assistant"
    content: str


@dataclass
class ProbeQuestion:
    question: str
    expected_answer: str
    # Which turn numbers contain the answer (for analysis)
    source_turns: list[int] = field(default_factory=list)
    # How many hops required to answer (1=single, 2=multi-hop, etc.)
    hops: int = 1


@dataclass
class TestCase:
    case_id: str
    dataset_type: DatasetType
    history: list[ConversationTurn]
    probes: list[ProbeQuestion]
    description: str = ""

    @property
    def n_turns(self) -> int:
        return len(self.history)


def get_support_ticket_dataset() -> list[TestCase]:
    """
    Long customer support conversations testing retention of:
    - User account details mentioned early
    - Technical constraints stated mid-conversation
    - Decisions and next steps agreed upon late
    """
    return [
        TestCase(
            case_id="support_001",
            dataset_type=DatasetType.SUPPORT_TICKET,
            description="Cloud migration support — 40 turns, facts spread across entire conversation",
            history=[
                ConversationTurn("user", "Hi, I'm Sarah Chen, account #4892. We're migrating our Django app to AWS."),
                ConversationTurn("assistant", "Hello Sarah! Happy to help with your AWS migration. What's your current setup?"),
                ConversationTurn("user", "We run Django 4.2 on Ubuntu 22.04. Our database is PostgreSQL 15 with about 200GB of data."),
                ConversationTurn("assistant", "Good setup. For a 200GB PostgreSQL database, I'd recommend RDS with a db.r6g.xlarge instance to start."),
                ConversationTurn("user", "We have a hard constraint: all data must stay in us-east-1 for compliance reasons."),
                ConversationTurn("assistant", "Understood — us-east-1 only. RDS supports that fully. Do you have a migration deadline?"),
                ConversationTurn("user", "Yes, we need to be fully migrated by March 15th. Our CTO is James Park."),
                ConversationTurn("assistant", "March 15th gives us about 8 weeks. That's tight but doable with the right plan."),
                ConversationTurn("user", "What about our Redis cache? We use Redis 7.0 for session management."),
                ConversationTurn("assistant", "ElastiCache for Redis is the natural fit. It handles Redis 7.0 natively."),
                ConversationTurn("user", "Our monthly budget cap for AWS is $3,000. Can we stay under that?"),
                ConversationTurn("assistant", "With your specs — RDS r6g.xlarge + ElastiCache — you'd be around $800-1,200/month. Well within budget."),
                ConversationTurn("user", "Great. What's the migration order you'd recommend?"),
                ConversationTurn("assistant", "I recommend: 1) Set up VPC, 2) RDS with read replica sync, 3) ElastiCache, 4) App servers, 5) DNS cutover."),
                ConversationTurn("user", "We also have 3 background workers using Celery with RabbitMQ."),
                ConversationTurn("assistant", "For Celery/RabbitMQ, Amazon MQ is the managed option. Alternatively, keep RabbitMQ on EC2."),
                ConversationTurn("user", "Let's use Amazon MQ. Can it handle our message volume? We peak at 5,000 messages/minute."),
                ConversationTurn("assistant", "Amazon MQ mq.m5.large handles well over 5,000 msg/min. You're fine there."),
                ConversationTurn("user", "What about SSL certificates? We have a wildcard cert for *.sarahco.com"),
                ConversationTurn("assistant", "Import your wildcard cert into ACM (AWS Certificate Manager) and it'll work across all services."),
                ConversationTurn("user", "We also need to maintain SOC 2 compliance throughout the migration."),
                ConversationTurn("assistant", "SOC 2 means: CloudTrail enabled, VPC flow logs, encryption at rest for RDS and EBS, and IAM least-privilege."),
                ConversationTurn("user", "Good. Let's confirm the final architecture decision: we're going with RDS r6g.xlarge?"),
                ConversationTurn("assistant", "Confirmed: RDS PostgreSQL 15 on db.r6g.xlarge, single-AZ to start, promote to Multi-AZ post-migration."),
                ConversationTurn("user", "One more thing — our staging environment uses a different DB: MySQL 8.0 with 40GB."),
                ConversationTurn("assistant", "For staging MySQL, RDS db.t3.medium would be cost-effective. Keep it in us-east-1 for consistency."),
                ConversationTurn("user", "Perfect. Can you summarise the key decisions we've made?"),
                ConversationTurn("assistant", "Summary: RDS r6g.xlarge (PostgreSQL 15), ElastiCache Redis, Amazon MQ, all in us-east-1, $800-1,200/month, March 15 deadline, SOC 2 compliant."),
                # Filler turns to stress memory
                ConversationTurn("user", "What monitoring would you recommend?"),
                ConversationTurn("assistant", "CloudWatch for metrics, with custom dashboards for RDS lag, ElastiCache hits, and MQ depth."),
                ConversationTurn("user", "Should we use Terraform or CloudFormation?"),
                ConversationTurn("assistant", "Terraform is more portable and has better state management. Given your team likely knows it, I'd stick with Terraform."),
                ConversationTurn("user", "What about CI/CD? We currently use GitHub Actions."),
                ConversationTurn("assistant", "GitHub Actions integrates well with AWS via OIDC — no long-lived credentials needed."),
                ConversationTurn("user", "Great. One last question about our database connection pooling."),
                ConversationTurn("assistant", "Use RDS Proxy — it handles connection pooling at the infrastructure level, reducing connection overhead for Django."),
            ],
            probes=[
                ProbeQuestion(
                    question="What is the user's account number?",
                    expected_answer="4892",
                    source_turns=[0],
                    hops=1,
                ),
                ProbeQuestion(
                    question="What is the migration deadline?",
                    expected_answer="March 15th",
                    source_turns=[6],
                    hops=1,
                ),
                ProbeQuestion(
                    question="What AWS region must all resources be in, and why?",
                    expected_answer="us-east-1, for compliance reasons",
                    source_turns=[4],
                    hops=1,
                ),
                ProbeQuestion(
                    question="What is the monthly AWS budget cap?",
                    expected_answer="$3,000",
                    source_turns=[10],
                    hops=1,
                ),
                ProbeQuestion(
                    question="What database instance type was chosen for the primary database, and what database version?",
                    expected_answer="db.r6g.xlarge running PostgreSQL 15",
                    source_turns=[3, 22],
                    hops=2,
                ),
                ProbeQuestion(
                    question="What is the peak message volume and what service was chosen to handle it?",
                    expected_answer="5,000 messages/minute, handled by Amazon MQ",
                    source_turns=[14, 16],
                    hops=2,
                ),
            ],
        ),
    ]


def get_all_test_cases() -> list[TestCase]:
    return [
        *get_support_ticket_dataset(),
    ]