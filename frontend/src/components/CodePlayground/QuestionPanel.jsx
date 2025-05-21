import QuestionGenerator from '../PracticeQuestion/QuestionGenerator';

/**
 * QuestionPanel now hosts the practice question generator and display.
 * This component embeds the full practice question workflow directly
 * into the code playground so users can generate questions without
 * leaving the page.
 */
function QuestionPanel() {
  return (
    <div className="panel question-panel">
      <div className="panel-header">Practice Questions</div>
      <div className="panel-content question-content">
        <QuestionGenerator />
      </div>
    </div>
  );
}

export default QuestionPanel;
