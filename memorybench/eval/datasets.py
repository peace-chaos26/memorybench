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

def get_research_assist_dataset() -> list[TestCase]:
    """
    Multi-session research assistant — tests user preference retention.
    Facts about research style, constraints, and prior conclusions
    must survive across many turns.
    """
    return [
        TestCase(
            case_id="research_001",
            dataset_type=DatasetType.RESEARCH_ASSIST,
            description="ML research assistant — 30 turns testing preference and fact retention",
            history=[
                ConversationTurn("user", "I'm Priya Mehta, a PhD student studying transformer attention mechanisms."),
                ConversationTurn("assistant", "Hi Priya! Happy to help with your transformer research. What's your focus?"),
                ConversationTurn("user", "I'm specifically looking at sparse attention patterns in long-context models. My advisor is Prof. Chen at MIT."),
                ConversationTurn("assistant", "Sparse attention is a great area. Longformer, BigBird, and FlashAttention all take different approaches."),
                ConversationTurn("user", "I need to avoid any papers that use proprietary datasets — my thesis has an open-source-only constraint."),
                ConversationTurn("assistant", "Understood — open datasets only. That rules out some GPT-4 analysis papers but leaves most academic work."),
                ConversationTurn("user", "My target conference is ICML 2025. Deadline is February 1st."),
                ConversationTurn("assistant", "ICML 2025 February deadline gives you about 4 months. That's tight for an empirical paper."),
                ConversationTurn("user", "We've already established that sliding window attention degrades on tasks requiring global context."),
                ConversationTurn("assistant", "Right — sliding window misses cross-document relationships. That's a known limitation of Longformer on multi-hop QA."),
                ConversationTurn("user", "My hypothesis is that learned sparse patterns outperform fixed patterns by at least 8% on SCROLLS benchmark."),
                ConversationTurn("assistant", "An 8% improvement claim on SCROLLS is ambitious but testable. What's your baseline?"),
                ConversationTurn("user", "Baseline is BigBird with block sparse attention. My model uses reinforcement learning to learn the sparsity mask."),
                ConversationTurn("assistant", "RL-learned sparsity masks are novel. The main risk is training instability — have you tried curriculum learning?"),
                ConversationTurn("user", "Yes, curriculum learning helped. Current results show 6.2% improvement, short of the 8% hypothesis."),
                ConversationTurn("assistant", "6.2% is still significant. Consider whether 8% was the right threshold or if SCROLLS subtasks vary."),
                ConversationTurn("user", "Good point. On the QA subtask specifically we see 9.1% improvement. Summarisation is dragging the average down."),
                ConversationTurn("assistant", "That's a strong result for QA. Frame it as task-specific: RL sparsity particularly benefits tasks requiring selective retrieval."),
                ConversationTurn("user", "We should also note I'm working in PyTorch, not JAX. Prof. Chen prefers PyTorch for reproducibility."),
                ConversationTurn("assistant", "PyTorch is the right call for reproducibility and community adoption. JAX has performance benefits but narrower audience."),
                ConversationTurn("user", "Can you help me structure the related work section?"),
                ConversationTurn("assistant", "Structure: (1) Fixed sparse attention, (2) Learned sparse attention, (3) Efficient attention broadly, (4) RL for architecture search."),
                ConversationTurn("user", "What should I cite for the RL for architecture search section?"),
                ConversationTurn("assistant", "Key citations: Neural Architecture Search (Zoph & Le 2017), DARTS (Liu et al 2019), and more recent work on differentiable sparsity."),
                ConversationTurn("user", "One constraint I forgot to mention — we can't use GPUs with more than 40GB VRAM due to lab resource limits."),
                ConversationTurn("assistant", "40GB VRAM cap means A100-40GB is your ceiling. That limits sequence length — have you profiled peak memory usage?"),
                ConversationTurn("user", "Peak is 38GB at sequence length 16384. So we're right at the limit."),
                ConversationTurn("assistant", "38GB at 16k tokens is tight. Gradient checkpointing could give you headroom if you need longer sequences."),
            ],
            probes=[
                ProbeQuestion(
                    question="What university is Priya's advisor at?",
                    expected_answer="MIT",
                    source_turns=[2], hops=1,
                ),
                ProbeQuestion(
                    question="What is the open-source constraint on this research?",
                    expected_answer="Cannot use proprietary datasets — open datasets only",
                    source_turns=[4], hops=1,
                ),
                ProbeQuestion(
                    question="What is the target conference and submission deadline?",
                    expected_answer="ICML 2025, deadline February 1st",
                    source_turns=[6], hops=1,
                ),
                ProbeQuestion(
                    question="What improvement did the model achieve on the QA subtask specifically?",
                    expected_answer="9.1% improvement on the QA subtask",
                    source_turns=[16], hops=1,
                ),
                ProbeQuestion(
                    question="What is the GPU VRAM constraint and what is the peak usage?",
                    expected_answer="40GB VRAM limit, peak usage is 38GB at sequence length 16384",
                    source_turns=[22, 24], hops=2,
                ),
            ],
        ),
    ]


def get_coding_session_dataset() -> list[TestCase]:
    """
    Debugging session — tests retention of technical constraints and
    decisions made mid-conversation under time pressure.
    """
    return [
        TestCase(
            case_id="coding_001",
            dataset_type=DatasetType.CODING_SESSION,
            description="Production bug debugging — 28 turns, technical facts spread throughout",
            history=[
                ConversationTurn("user", "We have a production incident. Our FastAPI service is throwing 503s. I'm Marcus, lead engineer at Stripe."),
                ConversationTurn("assistant", "Let's triage this fast. What's the error rate and when did it start?"),
                ConversationTurn("user", "Started 40 minutes ago. Error rate went from 0.1% to 34%. We're on Python 3.10, FastAPI 0.104."),
                ConversationTurn("assistant", "34% is severe. First check: is it all endpoints or specific routes?"),
                ConversationTurn("user", "Only the /payments/process endpoint. All others are fine. We deployed version 2.14.1 about an hour ago."),
                ConversationTurn("assistant", "Version 2.14.1 deployed one hour ago, issue started 40 minutes later — strong correlation. What changed in 2.14.1?"),
                ConversationTurn("user", "We added connection pooling to our PostgreSQL calls using asyncpg. Pool size set to 10."),
                ConversationTurn("assistant", "Pool size 10 with high concurrency is almost certainly the issue. What's your current RPS on /payments/process?"),
                ConversationTurn("user", "About 200 RPS. So we're exhausting the pool almost immediately."),
                ConversationTurn("assistant", "200 RPS with pool size 10 means each connection handles 20 req/s. If any query takes >50ms, you queue up instantly."),
                ConversationTurn("user", "Queries are averaging 180ms. So yes, exactly that problem."),
                ConversationTurn("assistant", "180ms × 200 RPS ÷ pool size 10 = 3,600 concurrent waits. Increase pool to at least 50, add pool_timeout."),
                ConversationTurn("user", "We can't go above 40 connections — our RDS instance is db.t3.medium with max_connections=100 and we have 3 app servers."),
                ConversationTurn("assistant", "3 servers × 40 connections = 120, over your 100 limit. Safe max per server is 30. Set pool size to 30."),
                ConversationTurn("user", "Changed to 30. Error rate is now 8%. Better but not resolved."),
                ConversationTurn("assistant", "8% remaining suggests query performance is also an issue. Can you EXPLAIN ANALYZE the slow payment query?"),
                ConversationTurn("user", "Found it — missing index on payments.user_id. Sequential scan on 4M rows."),
                ConversationTurn("assistant", "CREATE INDEX CONCURRENTLY on payments.user_id — CONCURRENTLY so it doesn't lock the table in production."),
                ConversationTurn("user", "Index created. Query time dropped from 180ms to 4ms. Error rate now 0.2%."),
                ConversationTurn("assistant", "0.2% residual — likely existing queued requests draining. Should clear in 2-3 minutes."),
                ConversationTurn("user", "Cleared. We're back to 0.1% baseline. Incident resolved. Total downtime was 55 minutes."),
                ConversationTurn("assistant", "Good resolution. Post-mortem items: add pool monitoring alerts, query performance tests in CI, index coverage checks."),
                ConversationTurn("user", "We should also note our SLA is 99.9% uptime. 55 minutes puts us over the monthly budget."),
                ConversationTurn("assistant", "55 minutes exceeds 99.9% SLA (43.8 min/month). You'll need to file an SLA credit report."),
            ],
            probes=[
                ProbeQuestion(
                    question="What version was deployed before the incident and when?",
                    expected_answer="Version 2.14.1, deployed about one hour before the incident started",
                    source_turns=[4], hops=1,
                ),
                ProbeQuestion(
                    question="What was the initial connection pool size and what was it changed to?",
                    expected_answer="Initial pool size was 10, changed to 30",
                    source_turns=[6, 12, 14], hops=2,
                ),
                ProbeQuestion(
                    question="What was the root cause of the remaining 8% errors after fixing the pool?",
                    expected_answer="Missing index on payments.user_id causing sequential scan on 4M rows",
                    source_turns=[16], hops=1,
                ),
                ProbeQuestion(
                    question="What is the company's SLA and did this incident violate it?",
                    expected_answer="99.9% uptime SLA, 55-minute downtime exceeded the monthly budget of 43.8 minutes",
                    source_turns=[20, 22], hops=2,
                ),
            ],
        ),
    ]

def get_all_test_cases() -> list[TestCase]:
    return [
        *get_support_ticket_dataset(),
        *get_research_assist_dataset(),
        *get_coding_session_dataset(),
    ]