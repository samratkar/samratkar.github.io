function addTask() {
    var taskInput = document.getElementById('new-task');
    var taskList = document.getElementById('task-list');
    if (taskInput.value) {
        var li = document.createElement('li');
        li.appendChild(document.createTextNode(taskInput.value));
        var deleteButton = document.createElement('button');
        deleteButton.appendChild(document.createTextNode('Delete'));
        deleteButton.onclick = function () {
            taskList.removeChild(li);
        };
        li.appendChild(deleteButton);
        taskList.appendChild(li);
        taskInput.value = '';
    }
}

function toggleTheme() {
    var app = document.getElementById('app');
    app.classList.toggle('dark-mode');
}


