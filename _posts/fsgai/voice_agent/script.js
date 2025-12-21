function addTask() {
    const input = document.getElementById('taskInput');
    const value = input.value.trim();
    if (!value) return;
    const li = document.createElement('li');
    li.textContent = value;
    li.onclick = function() { this.classList.toggle('completed'); };
    const del = document.createElement('button');
    del.textContent = 'Delete';
    del.className = 'delete-btn';
    del.onclick = function(e) {
        e.stopPropagation();
        li.remove();
    };
    li.appendChild(del);
    document.getElementById('taskList').appendChild(li);
    input.value = '';
}
document.getElementById('taskInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') addTask();
});
