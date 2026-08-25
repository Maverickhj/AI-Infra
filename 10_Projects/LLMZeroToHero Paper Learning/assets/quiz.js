document.querySelectorAll('.quiz').forEach((quiz) => {
  quiz.addEventListener('click', (event) => {
    const choice = event.target.closest('.choice');
    if (!choice || !quiz.contains(choice)) return;

    const feedback = quiz.querySelector('.feedback');
    const correct = choice.dataset.correct === 'true';
    feedback.className = 'feedback ' + (correct ? 'good' : 'try');
    feedback.textContent = correct
      ? quiz.dataset.correctFeedback
      : quiz.dataset.incorrectFeedback;
  });
});
