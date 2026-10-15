// The app works without JavaScript. These two helpers only improve the interface.
const taskSelect = document.querySelector('#task');

function updateModelFields() {
  const isClustering = taskSelect.value === 'clustering';
  document.querySelectorAll('[data-supervised]').forEach(function (field) {
    field.hidden = isClustering;
  });
  document.querySelectorAll('[data-clustering]').forEach(function (field) {
    field.hidden = !isClustering;
  });
}

if (taskSelect) {
  taskSelect.addEventListener('change', updateModelFields);
  updateModelFields();
}

// Disable only the submit button, so all the form's input values are still sent.
document.querySelectorAll('form[data-busy]').forEach(function (form) {
  form.addEventListener('submit', function () {
    const button = form.querySelector('button[type="submit"]');
    if (button) {
      button.disabled = true;
      button.textContent = form.dataset.busy;
      form.setAttribute('aria-busy', 'true');
    }
  });
});
