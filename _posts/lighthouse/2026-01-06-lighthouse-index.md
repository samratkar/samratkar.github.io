---
date: 2026-01-06 
layout: default
title: Lighthouse - Wisdom & Reflection
category: lighthouse
subcategory: lighthouse-index
tag: [lighthouse]
---

<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">

<style>
  body {
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
  }
  
  .gradient-bg {
    background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%);
  }
  
  .gradient-header {
    background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%) !important;
  }
  
  .gradient-header th {
    color: white !important;
    background: transparent !important;
  }
  
  .gradient-header tr {
    background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%) !important;
  }
  
  .side-panel {
    position: sticky;
    top: 1.5rem;
    max-height: calc(100vh - 3rem);
    overflow-y: auto;
  }
  
  .side-panel::-webkit-scrollbar {
    width: 6px;
  }
  
  .side-panel::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
  }
  
  .side-panel::-webkit-scrollbar-thumb {
    background: #9b87f5;
    border-radius: 10px;
  }
  
  .tag-item, .topic-item, .subtopic-item {
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .tag-item:hover, .topic-item:hover, .subtopic-item:hover {
    background-color: #f3e8ff;
    transform: translateX(4px);
  }
  
  .tag-item.active, .topic-item.active {
    background-color: #9b87f5;
    color: white;
    font-weight: 600;
  }
  
  .subtopic-item.active {
    background-color: #c4b5fd;
    color: white;
    font-weight: 600;
  }
  
  .topic-header {
    cursor: pointer;
    font-weight: 600;
  }
  
  .topic-section {
    margin-bottom: 0.75rem;
  }
  
  .subtopics-container {
    margin-left: 1rem;
    margin-top: 0.5rem;
    display: none;
  }
  
  .subtopics-container.expanded {
    display: block;
  }
  
  .article-row {
    transition: background-color 0.2s ease;
  }
  
  .article-row:hover {
    background-color: #faf5ff;
  }
  
  .hidden-row {
    display: none;
  }
  
  .stat-card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 4px 6px rgba(155, 135, 245, 0.15);
    transition: all 0.3s ease;
  }
  
  .stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 20px rgba(155, 135, 245, 0.2);
  }
  
  .stat-icon {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0;
    flex-shrink: 0;
  }
  
  .full-width-header {
    width: 100vw;
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
  }
  
  .full-width-container {
    width: 100vw;
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
  }
  
  .nav-menu {
    display: flex;
    gap: 1rem;
    align-items: center;
  }
  
  .nav-link {
    color: white;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    text-decoration: none;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.6rem 1.5rem;
    border-radius: 0.5rem;
    transition: all 0.3s ease;
    border: 2px solid rgba(255, 255, 255, 0.3);
  }
  
  .nav-link:hover {
    background: white;
    color: #9b87f5;
    border-color: white;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
  }
  
  @media (max-width: 768px) {
    .nav-menu {
      flex-wrap: wrap;
      gap: 0.5rem;
      justify-content: center;
    }
    
    .nav-link {
      font-size: 0.8rem;
      padding: 0.5rem 1rem;
    }
    
    .flex.gap-6 {
      flex-direction: column;
    }
    
    .w-72.flex-shrink-0 {
      width: 100% !important;
    }
  }
</style>

<div class="bg-gray-50 min-h-screen">
  <!-- Header - Full Width -->
  <header class="full-width-header gradient-bg text-white shadow-lg">
    <div class="max-w-screen-2xl mx-auto px-6 py-6">
      <div class="flex items-center justify-between flex-wrap gap-6">
        <div class="flex-1">
          <h1 class="text-3xl font-bold" style="font-family: 'Crimson Text', serif;">🌊 Lighthouse 🌊</h1>
          <p class="text-purple-100 mt-2 text-sm">Guiding lights through the storms of life — reflections on wisdom & philosophy</p>
        </div>
        <nav class="nav-menu">
          <a href="/" class="nav-link">🏠 Home</a>
          <a href="/about/" class="nav-link">👤 About</a>
          <a href="/books/" class="nav-link">📚 Books</a>
        </nav>
      </div>
    </div>
  </header>

{% assign lighthouse_posts = site.posts | where_exp: "post", "post.tags contains 'lighthouse'" %}
  {% assign all_tags = "" | split: "" %}
  {% assign all_topics = "" | split: "" %}
  {% for post in lighthouse_posts %}
    {% for tag in post.tags %}
      {% unless tag == "lighthouse" %}
        {% unless all_tags contains tag %}
          {% assign all_tags = all_tags | push: tag %}
        {% endunless %}
      {% endunless %}
    {% endfor %}
    {% if post.topic %}
      {% unless all_topics contains post.topic %}
        {% assign all_topics = all_topics | push: post.topic %}
      {% endunless %}
    {% endif %}
  {% endfor %}

  <!-- Statistics Section -->
  <div class="full-width-container bg-white border-b border-gray-200 shadow-sm">
    <div class="max-w-screen-2xl mx-auto px-6 py-6">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%);">
              <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
            </div>
            <div>
              <div class="text-2xl font-bold text-gray-800">{{ lighthouse_posts | size }}</div>
              <div class="text-sm text-gray-500">Total Articles</div>
            </div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
              <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"></path>
              </svg>
            </div>
            <div>
              <div class="text-2xl font-bold text-gray-800" id="tagCount">{{ all_tags | size }}</div>
              <div class="text-sm text-gray-500">Tags</div>
            </div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
              <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
              </svg>
            </div>
            <div>
              <div class="text-2xl font-bold text-gray-800" id="topicCount">{{ all_topics | size }}</div>
              <div class="text-sm text-gray-500">Topics</div>
            </div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);">
              <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"></path>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"></path>
              </svg>
            </div>
            <div>
              <div class="text-2xl font-bold text-gray-800" id="filteredCount">{{ lighthouse_posts | size }}</div>
              <div class="text-sm text-gray-500">Filtered Results</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Container - Full Width -->
  <div class="full-width-container">
    <div class="max-w-screen-2xl mx-auto px-6 py-6">
    
      <!-- Three Column Layout -->
      <div class="flex gap-6">
        
        <!-- Left Panel - Tags -->
        <div class="w-72 flex-shrink-0" style="width: 18rem;">
          <div class="side-panel bg-white rounded-xl shadow-lg p-4 border border-gray-100">
            <h3 class="text-base font-bold text-gray-800 mb-3 flex items-center gap-2 pb-2 border-b border-gray-200">
              <svg class="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"></path>
              </svg>
              <span>Tags</span>
            </h3>
            
            <div class="mb-4">
              <input 
                type="text" 
                id="tagSearch" 
                placeholder="Search tags..." 
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                onkeyup="filterTagsList()"
              >
            </div>
            
            <div class="mb-3">
              <button onclick="clearFilters()" class="w-full px-3 py-2 text-sm font-medium bg-gradient-to-r from-purple-500 to-purple-600 text-white hover:from-purple-600 hover:to-purple-700 rounded-lg transition shadow-sm">
                <svg class="w-4 h-4 inline-block mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                </svg>
                Show All Articles
              </button>
            </div>
            
            <div id="tagsList" class="space-y-1">
              {% assign sorted_tags = all_tags | sort_natural %}
              {% for tag in sorted_tags %}
              <div class="tag-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="filterByTag('{{ tag }}', this)" data-tag="{{ tag }}">
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                {{ tag | capitalize }}
              </div>
              {% endfor %}
            </div>
          </div>
        </div>
        
        <!-- Center - Articles Table -->
        <div class="flex-1 min-w-0">
          <div class="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-100">
            <div class="overflow-x-auto">
              <table class="w-full">
                <thead class="gradient-header" style="background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%) !important;">
                  <tr style="background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%) !important;">
                    <th class="px-4 py-3 text-left text-sm font-semibold" style="color: white !important; background: transparent !important; width: 60px;">#</th>
                    <th class="px-4 py-3 text-left text-sm font-semibold" style="color: white !important; background: transparent !important;">Title</th>
                    <th class="px-4 py-3 text-left text-sm font-semibold" style="color: white !important; background: transparent !important;">Summary</th>
                    <th class="px-4 py-3 text-left text-sm font-semibold" style="color: white !important; background: transparent !important;">Tags</th>
                  </tr>
                </thead>
                <tbody id="articlesTable" class="divide-y divide-gray-200">
                  {% for post in lighthouse_posts %}
                  {% assign tag_string = post.tags | join: "|" %}
                  {% assign subtopic_string = "" %}
                  {% if post.subtopics %}
                    {% assign subtopic_string = post.subtopics | join: "|" %}
                  {% endif %}
                  <tr class="article-row" 
                      data-tags="{{ tag_string }}"
                      {% if post.topic %}data-topic="{{ post.topic }}"{% endif %}
                      {% if subtopic_string != "" %}data-subtopics="{{ subtopic_string }}"{% endif %}>
                    <td class="px-4 py-3 text-sm text-gray-600 font-medium">{{ forloop.index }}</td>
                    <td class="px-4 py-3">
                      <a href="{{ post.url }}" target="_blank" class="text-purple-600 hover:text-purple-800 font-medium text-sm hover:underline">
                        {{ post.title }}
                      </a>
                    </td>
                    <td class="px-4 py-3">
                      <div class="text-sm text-gray-600 line-clamp-2">
                        {% if post.summary %}
                          {{ post.summary }}
                        {% elsif post.excerpt %}
                          {{ post.excerpt | strip_html | truncatewords: 30 }}
                        {% else %}
                          {{ post.content | strip_html | truncatewords: 30 }}
                        {% endif %}
                      </div>
                    </td>
                    <td class="px-4 py-3">
                      <div class="flex flex-wrap gap-1">
                        {% for tag in post.tags %}
                          <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-700">{{ tag }}</span>
                        {% endfor %}
                      </div>
                    </td>
                  </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        
        <!-- Right Panel - Topics & Sub-Topics -->
        <div class="w-72 flex-shrink-0" style="width: 18rem;">
          <div class="side-panel bg-white rounded-xl shadow-lg p-4 border border-gray-100">
            <h3 class="text-base font-bold text-gray-800 mb-3 flex items-center gap-2 pb-2 border-b border-gray-200">
              <svg class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
              </svg>
              <span>Topics</span>
            </h3>
            
            <div class="space-y-1">
              {% for topic in all_topics %}
                {% assign topic_id = topic | slugify %}
                {% assign topic_posts = lighthouse_posts | where: "topic", topic %}
                {% if topic_posts.size > 0 %}
              <div class="topic-section">
                <div class="topic-item topic-header px-3 py-2.5 rounded-lg text-sm flex items-center justify-between" onclick="toggleTopic('{{ topic_id }}', this)" data-topic="{{ topic }}">
                  <div>
                    <span class="inline-block w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                    <span>{{ topic }}</span>
                  </div>
                  <svg class="w-4 h-4 transform transition-transform topic-chevron" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
                <div id="{{ topic_id }}-subtopics" class="subtopics-container">
                  {% for post in topic_posts %}
                    {% if post.subtopics %}
                      {% for subtopic in post.subtopics %}
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('{{ subtopic }}', this)" data-subtopic="{{ subtopic }}">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    {{ subtopic }}
                  </div>
                      {% endfor %}
                    {% endif %}
                  {% endfor %}
                </div>
              </div>
                {% endif %}
              {% endfor %}
            </div>
          </div>
        </div>
        
      </div>
    </div>
  </div>
  
  <!-- Footer Quote -->
  <div class="full-width-container gradient-bg py-12">
    <div class="text-center px-6">
      <p class="text-white text-2xl font-serif italic" style="font-family: 'Crimson Text', serif;">
        "Be like the lighthouse that guides through darkness, unwavering in its purpose."
      </p>
    </div>
  </div>
</div>

<script>
  let activeFilter = { type: null, value: null };
  
  function updateFilteredCount() {
    const visibleRows = document.querySelectorAll('.article-row:not(.hidden-row)').length;
    document.getElementById('filteredCount').textContent = visibleRows;
  }
  
  function filterByTag(tag, element) {
    const tagItems = document.querySelectorAll('.tag-item');
    const topicItems = document.querySelectorAll('.topic-item');
    const subtopicItems = document.querySelectorAll('.subtopic-item');
    
    tagItems.forEach(item => item.classList.remove('active'));
    topicItems.forEach(item => item.classList.remove('active'));
    subtopicItems.forEach(item => item.classList.remove('active'));
    
    element.classList.add('active');
    activeFilter = { type: 'tag', value: tag };
    
    const rows = document.querySelectorAll('.article-row');
    rows.forEach(row => {
      const tags = row.dataset.tags ? row.dataset.tags.split('|') : [];
      if (tags.includes(tag)) {
        row.classList.remove('hidden-row');
      } else {
        row.classList.add('hidden-row');
      }
    });
    
    updateFilteredCount();
  }
  
  function toggleTopic(topicId, element) {
    const subtopicsContainer = document.getElementById(topicId + '-subtopics');
    const chevron = element.querySelector('.topic-chevron');
    
    if (subtopicsContainer.classList.contains('expanded')) {
      subtopicsContainer.classList.remove('expanded');
      chevron.style.transform = 'rotate(0deg)';
    } else {
      // Close all other topics
      document.querySelectorAll('.subtopics-container').forEach(container => {
        container.classList.remove('expanded');
      });
      document.querySelectorAll('.topic-chevron').forEach(chev => {
        chev.style.transform = 'rotate(0deg)';
      });
      
      subtopicsContainer.classList.add('expanded');
      chevron.style.transform = 'rotate(180deg)';
    }
    
    // Also filter by topic when clicking
    const topic = element.dataset.topic;
    filterByTopic(topic, element);
  }
  
  function filterByTopic(topic, element) {
    const tagItems = document.querySelectorAll('.tag-item');
    const topicItems = document.querySelectorAll('.topic-item');
    const subtopicItems = document.querySelectorAll('.subtopic-item');
    
    tagItems.forEach(item => item.classList.remove('active'));
    topicItems.forEach(item => item.classList.remove('active'));
    subtopicItems.forEach(item => item.classList.remove('active'));
    
    element.classList.add('active');
    activeFilter = { type: 'topic', value: topic };
    
    const rows = document.querySelectorAll('.article-row');
    rows.forEach(row => {
      const rowTopic = row.dataset.topic || '';
      if (rowTopic === topic) {
        row.classList.remove('hidden-row');
      } else {
        row.classList.add('hidden-row');
      }
    });
    
    updateFilteredCount();
  }
  
  function filterBySubtopic(subtopic, element) {
    const tagItems = document.querySelectorAll('.tag-item');
    const topicItems = document.querySelectorAll('.topic-item');
    const subtopicItems = document.querySelectorAll('.subtopic-item');
    
    tagItems.forEach(item => item.classList.remove('active'));
    topicItems.forEach(item => item.classList.remove('active'));
    subtopicItems.forEach(item => item.classList.remove('active'));
    
    element.classList.add('active');
    activeFilter = { type: 'subtopic', value: subtopic };
    
    const rows = document.querySelectorAll('.article-row');
    rows.forEach(row => {
      const subtopics = row.dataset.subtopics ? row.dataset.subtopics.split('|') : [];
      if (subtopics.includes(subtopic)) {
        row.classList.remove('hidden-row');
      } else {
        row.classList.add('hidden-row');
      }
    });
    
    updateFilteredCount();
  }
  
  function clearFilters() {
    const tagItems = document.querySelectorAll('.tag-item');
    const topicItems = document.querySelectorAll('.topic-item');
    const subtopicItems = document.querySelectorAll('.subtopic-item');
    const rows = document.querySelectorAll('.article-row');
    
    tagItems.forEach(item => item.classList.remove('active'));
    topicItems.forEach(item => item.classList.remove('active'));
    subtopicItems.forEach(item => item.classList.remove('active'));
    rows.forEach(row => row.classList.remove('hidden-row'));
    
    activeFilter = { type: null, value: null };
    updateFilteredCount();
  }
  
  function filterTagsList() {
    const searchInput = document.getElementById('tagSearch').value.toLowerCase();
    const tagItems = document.querySelectorAll('.tag-item');
    
    tagItems.forEach(item => {
      const tagText = item.textContent.toLowerCase().trim();
      if (tagText.includes(searchInput)) {
        item.style.display = '';
      } else {
        item.style.display = 'none';
      }
    });
  }
</script>
