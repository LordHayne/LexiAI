"""
Umfassende Tests für Phase 3: Self-Correction System

Testet:
1. SelfCorrectionAnalyzer - Fehleranalyse
2. SelfCorrectionAnalyzer - Korrektur-Generierung
3. Correction Memory Erstellung
4. analyze_and_correct_failures() - Heartbeat Integration
5. End-to-End Self-Correction Flow
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime, timezone
from uuid import uuid4


class TestSelfCorrectionAnalyzer:
    """Tests für SelfCorrectionAnalyzer Klasse."""

    def test_analyze_failure_factually_wrong(self):
        """Teste Fehleranalyse für faktisch falsche Antwort."""
        from backend.memory.self_correction import SelfCorrectionAnalyzer
        from backend.models.feedback import ConversationTurn, FeedbackEntry, FeedbackType, ErrorCategory

        # Mock LLM
        mock_chat_client = Mock()
        mock_response = Mock()
        mock_response.content = """ERROR_CATEGORY: FACTUALLY_WRONG
ANALYSIS:
Die Antwort enthält falsche Informationen. Docker wurde 2013 veröffentlicht, nicht 2010.
Der Fehler liegt darin, dass veraltete oder falsche Fakten verwendet wurden."""
        mock_chat_client.invoke = Mock(return_value=mock_response)

        mock_embeddings = Mock()

        analyzer = SelfCorrectionAnalyzer(mock_chat_client, mock_embeddings)

        # Erstelle fehlerhaften Turn
        turn = ConversationTurn(
            turn_id=str(uuid4()),
            user_id="test_user",
            user_message="When was Docker released?",
            ai_response="Docker was released in 2010.",
            timestamp=datetime.now(timezone.utc)
        )

        feedback = FeedbackEntry(
            feedback_id=str(uuid4()),
            turn_id=turn.turn_id,
            feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
            timestamp=datetime.now(timezone.utc)
        )

        # Analysiere Fehler
        error_category, analysis = analyzer.analyze_failure(turn, [feedback])

        # Assertions
        assert error_category == ErrorCategory.FACTUALLY_WRONG
        assert "falsche" in analysis.lower() or "wrong" in analysis.lower()
        assert mock_chat_client.invoke.called

        print("✅ Test: analyze_failure für FACTUALLY_WRONG")

    def test_analyze_failure_incomplete(self):
        """Teste Fehleranalyse für unvollständige Antwort."""
        from backend.memory.self_correction import SelfCorrectionAnalyzer
        from backend.models.feedback import ConversationTurn, FeedbackEntry, FeedbackType, ErrorCategory

        # Mock LLM mit INCOMPLETE Antwort
        mock_chat_client = Mock()
        mock_response = Mock()
        mock_response.content = """ERROR_CATEGORY: INCOMPLETE
ANALYSIS:
Die Antwort erklärt nur was Docker ist, aber nicht WIE man es installiert.
Wichtige Schritte fehlen komplett."""
        mock_chat_client.invoke = Mock(return_value=mock_response)

        mock_embeddings = Mock()

        analyzer = SelfCorrectionAnalyzer(mock_chat_client, mock_embeddings)

        turn = ConversationTurn(
            turn_id=str(uuid4()),
            user_id="test_user",
            user_message="How do I install Docker?",
            ai_response="Docker is a containerization platform.",
            timestamp=datetime.now(timezone.utc)
        )

        feedback = FeedbackEntry(
            feedback_id=str(uuid4()),
            turn_id=turn.turn_id,
            feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
            user_comment="Doesn't answer my question",
            timestamp=datetime.now(timezone.utc)
        )

        error_category, analysis = analyzer.analyze_failure(turn, [feedback])

        assert error_category == ErrorCategory.INCOMPLETE
        assert "incomplete" in analysis.lower() or "unvollständig" in analysis.lower() or "fehlen" in analysis.lower()

        print("✅ Test: analyze_failure für INCOMPLETE")

    def test_generate_correction(self):
        """Teste Korrektur-Generierung."""
        from backend.memory.self_correction import SelfCorrectionAnalyzer
        from backend.models.feedback import ConversationTurn, ErrorCategory

        # Mock LLM
        mock_chat_client = Mock()
        mock_response = Mock()
        mock_response.content = """Docker was released in March 2013 by Solomon Hykes.
It revolutionized containerization and became widely adopted for application deployment.

To install Docker:
1. Visit docker.com
2. Download Docker Desktop for your OS
3. Run the installer
4. Verify installation with: docker --version"""
        mock_chat_client.invoke = Mock(return_value=mock_response)

        mock_embeddings = Mock()

        analyzer = SelfCorrectionAnalyzer(mock_chat_client, mock_embeddings)

        turn = ConversationTurn(
            turn_id=str(uuid4()),
            user_id="test_user",
            user_message="When was Docker released and how do I install it?",
            ai_response="Docker is a container platform.",
            timestamp=datetime.now(timezone.utc)
        )

        analysis = "Die Antwort ist unvollständig und beantwortet die Frage nicht vollständig."

        correction = analyzer.generate_correction(
            turn,
            ErrorCategory.INCOMPLETE,
            analysis
        )

        # Assertions
        assert correction is not None
        assert len(correction) > 0
        assert "2013" in correction or "install" in correction.lower()
        assert mock_chat_client.invoke.called

        print("✅ Test: generate_correction")

    def test_create_correction_memory(self):
        """Teste Erstellung einer Correction Memory."""
        from backend.memory.self_correction import SelfCorrectionAnalyzer
        from backend.models.feedback import ConversationTurn, ErrorCategory

        # Mock Embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_query = Mock(return_value=[0.1] * 768)

        mock_chat_client = Mock()

        analyzer = SelfCorrectionAnalyzer(mock_chat_client, mock_embeddings)

        turn = ConversationTurn(
            turn_id=str(uuid4()),
            user_id="test_user",
            user_message="What is Kubernetes?",
            ai_response="It's a Google product.",
            timestamp=datetime.now(timezone.utc)
        )

        correction = "Kubernetes is an open-source container orchestration platform..."
        analysis = "Antwort war zu simpel und ungenau."

        correction_memory = analyzer.create_correction_memory(
            turn,
            correction,
            ErrorCategory.TOO_SIMPLE,
            analysis
        )

        # Assertions
        assert correction_memory is not None
        assert correction_memory.category == "self_correction"
        assert correction_memory.relevance == 1.0  # Hohe Relevanz!
        assert "correction" in [tag.lower() for tag in correction_memory.tags]
        assert "SELBST-KORREKTUR" in correction_memory.content or "SELF-CORRECTION" in correction_memory.content
        assert turn.user_message in correction_memory.content
        assert correction in correction_memory.content
        assert len(correction_memory.embedding) == 768

        print("✅ Test: create_correction_memory")

    def test_extract_error_category_from_analysis(self):
        """Teste ErrorCategory Extraktion aus LLM-Antwort."""
        from backend.memory.self_correction import SelfCorrectionAnalyzer
        from backend.models.feedback import ErrorCategory

        analyzer = SelfCorrectionAnalyzer(Mock(), Mock())

        # Test verschiedene Formate
        test_cases = [
            ("ERROR_CATEGORY: FACTUALLY_WRONG\nANALYSIS: ...", ErrorCategory.FACTUALLY_WRONG),
            ("ERROR_CATEGORY: INCOMPLETE\nSomething...", ErrorCategory.INCOMPLETE),
            ("ERROR_CATEGORY: HALLUCINATION\n...", ErrorCategory.HALLUCINATION),
            ("ERROR_CATEGORY: MISSING_CONTEXT\n...", ErrorCategory.MISSING_CONTEXT),
            ("ERROR_CATEGORY: TOO_TECHNICAL\n...", ErrorCategory.TOO_TECHNICAL),
            ("ERROR_CATEGORY: TOO_SIMPLE\n...", ErrorCategory.TOO_SIMPLE),
            ("ERROR_CATEGORY: IRRELEVANT\n...", ErrorCategory.IRRELEVANT),
            ("No category found", ErrorCategory.INCOMPLETE),  # Fallback
        ]

        for analysis_text, expected_category in test_cases:
            result = analyzer._extract_error_category(analysis_text)
            assert result == expected_category, f"Failed for: {analysis_text}"

        print("✅ Test: extract_error_category")


class TestAnalyzeAndCorrectFailures:
    """Tests für die Heartbeat-Integration Funktion."""

    def test_analyze_and_correct_failures_no_negative_turns(self):
        """Teste wenn keine negativen Turns vorhanden sind."""
        from backend.memory.self_correction import analyze_and_correct_failures
        from backend.memory.conversation_tracker import ConversationTracker

        # Mock ConversationTracker
        with patch('backend.memory.conversation_tracker.get_conversation_tracker') as mock_get_tracker:
            mock_tracker = Mock()
            mock_tracker.get_negative_turns = Mock(return_value=[])
            mock_get_tracker.return_value = mock_tracker

            # Mock Components
            with patch('backend.core.component_cache.get_cached_components') as mock_get_components:
                mock_bundle = Mock()
                mock_get_components.return_value = mock_bundle

                corrections_count = analyze_and_correct_failures()

                assert corrections_count == 0
                assert mock_tracker.get_negative_turns.called

        print("✅ Test: analyze_and_correct_failures ohne negative Turns")

    def test_analyze_and_correct_failures_with_stop_signal(self):
        """Teste Unterbrechung durch Stop-Signal."""
        from backend.memory.self_correction import analyze_and_correct_failures
        from backend.models.feedback import ConversationTurn, FeedbackEntry, FeedbackType

        # Simuliere viele negative Turns
        negative_turns = []
        for i in range(10):
            turn = ConversationTurn(
                turn_id=f"turn_{i}",
                user_id="test_user",
                user_message=f"Question {i}",
                ai_response=f"Bad answer {i}",
                timestamp=datetime.now(timezone.utc)
            )
            feedback = FeedbackEntry(
                feedback_id=f"fb_{i}",
                turn_id=turn.turn_id,
                feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
                timestamp=datetime.now(timezone.utc)
            )
            negative_turns.append((turn, [feedback]))

        with patch('backend.memory.conversation_tracker.get_conversation_tracker') as mock_get_tracker:
            mock_tracker = Mock()
            mock_tracker.get_negative_turns = Mock(return_value=negative_turns)
            mock_get_tracker.return_value = mock_tracker

            with patch('backend.core.component_cache.get_cached_components') as mock_get_components:
                mock_bundle = Mock()
                mock_bundle.vectorstore = Mock()
                mock_bundle.chat_client = Mock()
                mock_bundle.embeddings = Mock()
                mock_get_components.return_value = mock_bundle

                # Stop-Signal nach 2 Corrections
                stop_counter = [0]
                def stop_check():
                    stop_counter[0] += 1
                    return stop_counter[0] > 2

                corrections_count = analyze_and_correct_failures(stop_check_fn=stop_check)

                # Sollte nach ~2-3 Corrections gestoppt haben
                assert corrections_count <= 3

        print("✅ Test: analyze_and_correct_failures mit Stop-Signal")


class TestEndToEndSelfCorrection:
    """End-to-End Tests für kompletten Self-Correction Flow."""

    def test_complete_self_correction_flow(self):
        """
        Kompletter E2E Test:
        1. User stellt Frage
        2. System antwortet schlecht
        3. User gibt negatives Feedback
        4. Heartbeat erkennt Fehler
        5. System analysiert und korrigiert
        6. Correction Memory wird erstellt
        7. Nächste ähnliche Frage verwendet Correction
        """
        from backend.memory.conversation_tracker import get_conversation_tracker
        from backend.memory.self_correction import SelfCorrectionAnalyzer
        from backend.models.feedback import FeedbackType, ErrorCategory

        tracker = get_conversation_tracker()

        # 1. Schlechte Antwort
        turn_id = tracker.record_turn(
            user_id="e2e_user",
            user_message="How do I exit vim?",
            ai_response="Just close the terminal."
        )

        # 2. Negatives Feedback
        tracker.record_feedback(
            turn_id,
            FeedbackType.EXPLICIT_NEGATIVE,
            user_comment="That doesn't help, I lose my work!"
        )

        # 3. Mock LLM für Analyse und Korrektur
        mock_chat_client = Mock()

        # Erste Call: Analyse
        analysis_response = Mock()
        analysis_response.content = """ERROR_CATEGORY: INCOMPLETE
ANALYSIS:
Die Antwort ist nicht hilfreich. Terminal schließen führt zu Datenverlust.
Die korrekte Antwort sollte :q oder :wq erklären."""

        # Zweite Call: Korrektur
        correction_response = Mock()
        correction_response.content = """To exit vim:
- :q - quit (if no changes)
- :q! - quit without saving
- :wq - save and quit
- :x - save and quit (shorter)"""

        mock_chat_client.invoke = Mock(side_effect=[analysis_response, correction_response])

        mock_embeddings = Mock()
        mock_embeddings.embed_query = Mock(return_value=[0.1] * 768)

        # 4. Analysiere und korrigiere
        analyzer = SelfCorrectionAnalyzer(mock_chat_client, mock_embeddings)

        turn = tracker.get_turn(turn_id)
        feedbacks = tracker.get_feedback_for_turn(turn_id)

        error_category, analysis = analyzer.analyze_failure(turn, feedbacks)
        assert error_category == ErrorCategory.INCOMPLETE

        correction = analyzer.generate_correction(turn, error_category, analysis)
        assert ":q" in correction or ":wq" in correction

        # 5. Erstelle Correction Memory
        correction_memory = analyzer.create_correction_memory(
            turn, correction, error_category, analysis
        )

        assert correction_memory.category == "self_correction"
        assert correction_memory.relevance == 1.0
        assert "vim" in correction_memory.content.lower()

        # 6. Simuliere dass Correction Memory im Retrieval gefunden wird
        # Bei nächster ähnlicher Frage sollte diese Memory bevorzugt werden
        # (Das wird durch Prioritization in chat_processing.py sichergestellt)

        print("✅ Test: Kompletter E2E Self-Correction Flow")

    def test_multiple_feedbacks_same_turn(self):
        """Teste mehrere Feedbacks für denselben Turn."""
        from backend.memory.conversation_tracker import get_conversation_tracker
        from backend.models.feedback import FeedbackType

        tracker = get_conversation_tracker()

        turn_id = tracker.record_turn(
            user_id="multi_fb_user",
            user_message="Test question",
            ai_response="Test answer"
        )

        # Mehrere Feedbacks
        tracker.record_feedback(turn_id, FeedbackType.EXPLICIT_NEGATIVE)
        tracker.record_feedback(
            turn_id,
            FeedbackType.EXPLICIT_NEGATIVE,
            user_comment="Wrong answer"
        )

        feedbacks = tracker.get_feedback_for_turn(turn_id)

        assert len(feedbacks) == 2
        assert all(fb.feedback_type == FeedbackType.EXPLICIT_NEGATIVE for fb in feedbacks)

        print("✅ Test: Mehrere Feedbacks für einen Turn")

    def test_correction_memory_high_relevance(self):
        """Teste dass Correction Memories hohe Relevance haben."""
        from backend.memory.self_correction import SelfCorrectionAnalyzer
        from backend.models.feedback import ConversationTurn, ErrorCategory

        mock_embeddings = Mock()
        mock_embeddings.embed_query = Mock(return_value=[0.1] * 768)

        analyzer = SelfCorrectionAnalyzer(Mock(), mock_embeddings)

        turn = ConversationTurn(
            turn_id=str(uuid4()),
            user_id="test_user",
            user_message="Test",
            ai_response="Test",
            timestamp=datetime.now(timezone.utc)
        )

        correction_memory = analyzer.create_correction_memory(
            turn,
            "Corrected answer",
            ErrorCategory.FACTUALLY_WRONG,
            "Analysis"
        )

        # Correction Memories sollten immer Relevanz 1.0 haben
        assert correction_memory.relevance == 1.0
        assert correction_memory.category == "self_correction"
        assert "learning" in correction_memory.tags

        print("✅ Test: Correction Memory hat hohe Relevance")


class TestFeedbackTypes:
    """Tests für verschiedene Feedback-Typen."""

    def test_explicit_positive_feedback(self):
        """Teste explizit positives Feedback."""
        from backend.memory.conversation_tracker import get_conversation_tracker
        from backend.models.feedback import FeedbackType

        tracker = get_conversation_tracker()

        turn_id = tracker.record_turn(
            user_id="positive_user",
            user_message="Great question",
            ai_response="Great answer"
        )

        tracker.record_feedback(
            turn_id,
            FeedbackType.EXPLICIT_POSITIVE,
            user_comment="Very helpful!"
        )

        feedbacks = tracker.get_feedback_for_turn(turn_id)

        assert len(feedbacks) == 1
        assert feedbacks[0].feedback_type == FeedbackType.EXPLICIT_POSITIVE
        assert feedbacks[0].user_comment == "Very helpful!"

        # Positive Turns sollten NICHT in get_negative_turns() erscheinen
        negative_turns = tracker.get_negative_turns()
        assert not any(turn.turn_id == turn_id for turn, _ in negative_turns)

        print("✅ Test: Explizit positives Feedback")

    def test_implicit_reformulation_feedback(self):
        """Teste implizites Reformulation Feedback."""
        from backend.memory.conversation_tracker import get_conversation_tracker
        from backend.models.feedback import FeedbackType

        tracker = get_conversation_tracker()

        turn_id = tracker.record_turn(
            user_id="implicit_user",
            user_message="Original question",
            ai_response="Response"
        )

        # Implicit Reformulation (automatisch erkannt)
        tracker.record_feedback(
            turn_id,
            FeedbackType.IMPLICIT_REFORMULATION,
            confidence=0.75
        )

        feedbacks = tracker.get_feedback_for_turn(turn_id)

        assert len(feedbacks) == 1
        assert feedbacks[0].feedback_type == FeedbackType.IMPLICIT_REFORMULATION
        assert feedbacks[0].confidence == 0.75

        print("✅ Test: Implizites Reformulation Feedback")


def run_all_tests():
    """Führt alle Tests aus."""
    print("\n🧪 Testing Phase 3: Self-Correction System...\n")

    test_classes = [
        TestSelfCorrectionAnalyzer,
        TestAnalyzeAndCorrectFailures,
        TestEndToEndSelfCorrection,
        TestFeedbackTypes
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n📦 {test_class.__name__}:")
        test_instance = test_class()

        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    passed_tests += 1
                except Exception as e:
                    print(f"   ❌ {method_name}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"✅ {passed_tests}/{total_tests} Tests bestanden")

    if passed_tests == total_tests:
        print("\n🎉 Alle Self-Correction Tests erfolgreich!")
        print("\n📋 Phase 3.9 komplett:")
        print("   ✅ Fehleranalyse Tests")
        print("   ✅ Korrektur-Generierung Tests")
        print("   ✅ Correction Memory Tests")
        print("   ✅ Heartbeat Integration Tests")
        print("   ✅ End-to-End Flow Tests")
        print("   ✅ Feedback-Typen Tests")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} Tests fehlgeschlagen")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
