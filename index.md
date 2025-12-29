---
layout: default
title: Home - All Articles
---

<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
  body {
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
  }
  
  .gradient-bg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }
  
  .gradient-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  }
  
  .gradient-header th {
    color: white !important;
    background: transparent !important;
  }
  
  .gradient-header tr {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  }
  
  table thead.gradient-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  }
  
  .glass-effect {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
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
    background: #667eea;
    border-radius: 10px;
  }
  
  .tag-item, .category-item {
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .tag-item:hover, .category-item:hover {
    background-color: #e0e7ff;
    transform: translateX(4px);
  }
  
  .tag-item.active, .category-item.active {
    background-color: #667eea;
    color: white;
    font-weight: 600;
  }
  
  .category-parent {
    cursor: pointer;
    user-select: none;
  }
  
  .category-parent .expand-icon {
    transition: transform 0.2s ease;
    display: inline-block;
  }
  
  .category-parent.expanded .expand-icon {
    transform: rotate(90deg);
  }
  
  .subcategory-list {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
  }
  
  .subcategory-list.expanded {
    max-height: 1000px;
  }
  
  .subcategory-item {
    margin-left: 1.25rem;
    font-size: 0.813rem;
  }
  
  .article-row {
    transition: background-color 0.2s ease;
  }
  
  .article-row:hover {
    background-color: #f9fafb;
  }
  
  .hidden-row {
    display: none;
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
  
  .stat-card {
    background: white;
    border-radius: 10px;
    padding: 0.75rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    transition: all 0.3s ease;
  }
  
  .stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 20px rgba(0, 0, 0, 0.1);
  }
  
  .stat-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0;
    flex-shrink: 0;
  }
  
  /* Hide default theme header */
  .site-header {
    display: none !important;
  }
  
  header.site-header {
    display: none !important;
  }
</style>

<div class="bg-gray-50 min-h-screen">
  <!-- Header - Full Width -->
  <header class="full-width-header gradient-bg text-white shadow-lg">
    <div class="max-w-screen-2xl mx-auto px-6 py-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold">Samrat Kar | exploring & experimenting</h1>
          <p class="text-blue-100 mt-2 text-sm">Article Index - All my writings and explorations</p>
        </div>
        <div class="text-right">
          <div class="text-xl font-bold">{{ site.posts | size }}</div>
          <div class="text-xs text-blue-100">Total Articles</div>
        </div>
      </div>
    </div>
  </header>

  <!-- Collect all tags and categories from all posts FIRST -->
  {% assign all_tags = "" | split: "" %}
  {% assign all_categories = "" | split: "" %}
  {% for post in site.posts %}
    <!-- Collect from both 'tags' and 'tag' fields -->
    {% for tag in post.tags %}
      {% assign tag_str = tag | append: "" %}
      {% unless all_tags contains tag_str %}
        {% assign all_tags = all_tags | push: tag_str %}
      {% endunless %}
    {% endfor %}
    {% if post.tag %}
      {% assign tag_value = post.tag | append: "" %}
      {% unless all_tags contains tag_value %}
        {% assign all_tags = all_tags | push: tag_value %}
      {% endunless %}
    {% endif %}
    <!-- Collect categories -->
    {% for category in post.categories %}
      {% assign cat_str = category | append: "" %}
      {% unless all_categories contains cat_str %}
        {% assign all_categories = all_categories | push: cat_str %}
      {% endunless %}
    {% endfor %}
  {% endfor %}
  {% assign all_tags = all_tags | sort %}
  {% assign all_categories = all_categories | sort %}

  <!-- Statistics Section -->
  <div class="full-width-container bg-white border-b border-gray-200 shadow-sm">
    <div class="max-w-screen-2xl mx-auto px-6 py-6">
      {% assign total_posts = site.posts | size %}
      {% assign posts_with_tags = 0 %}
      {% assign posts_with_categories = 0 %}
      {% assign total_tags_count = 0 %}
      {% assign total_categories_count = 0 %}
      
      {% for post in site.posts %}
        {% if post.tags.size > 0 or post.tag %}
          {% assign posts_with_tags = posts_with_tags | plus: 1 %}
        {% endif %}
        {% if post.categories.size > 0 %}
          {% assign posts_with_categories = posts_with_categories | plus: 1 %}
        {% endif %}
        {% assign total_tags_count = total_tags_count | plus: post.tags.size %}
        {% assign total_categories_count = total_categories_count | plus: post.categories.size %}
      {% endfor %}
      
      <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
              <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
            </div>
            <div>
              <div class="text-lg font-bold text-gray-800">{{ total_posts }}</div>
              <div class="text-xs text-gray-500">Total Articles</div>
            </div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);">
              <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"></path>
              </svg>
            </div>
            <div>
              <div class="text-lg font-bold text-gray-800">{{ all_tags.size }}</div>
              <div class="text-xs text-gray-500">Unique Tags</div>
            </div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
              <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
              </svg>
            </div>
            <div>
              <div class="text-lg font-bold text-gray-800">{{ all_categories.size }}</div>
              <div class="text-xs text-gray-500">Categories</div>
            </div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
              <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"></path>
              </svg>
            </div>
            <div>
              <div class="text-lg font-bold text-gray-800">{{ posts_with_tags }}</div>
              <div class="text-xs text-gray-500">Tagged Articles</div>
            </div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
              <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"></path>
              </svg>
            </div>
            <div>
              <div class="text-lg font-bold text-gray-800">{{ posts_with_categories }}</div>
              <div class="text-xs text-gray-500">Categorized</div>
            </div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);">
              <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"></path>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"></path>
              </svg>
            </div>
            <div>
              <div class="text-lg font-bold text-gray-800" id="filteredCountTop">{{ total_posts }}</div>
              <div class="text-xs text-gray-500">Filtered Results</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Container - Full Width -->
  <div class="full-width-container">
    <div class="max-w-screen-2xl mx-auto px-6 py-6">
    
    {% assign sorted_posts = site.posts | sort: 'date' | reverse %}
    
    <!-- Three Column Layout -->
    <div class="flex gap-6">
      
      <!-- Left Panel - Tags -->
      <div class="w-72 flex-shrink-0">
        <div class="side-panel bg-white rounded-xl shadow-lg p-4 border border-gray-100">
          <h3 class="text-base font-bold text-gray-800 mb-3 flex items-center gap-2 pb-2 border-b border-gray-200">
            <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"></path>
            </svg>
            <span>Tags</span>
            <span class="ml-auto text-sm font-normal text-gray-500">({{ all_tags.size }})</span>
          </h3>
          
          <div class="mb-4">
            <input 
              type="text" 
              id="tagSearch" 
              placeholder="Search tags..." 
              class="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              onkeyup="filterTagsList()"
            >
          </div>
          
          <div class="mb-3">
            <button onclick="clearFilters()" class="w-full px-3 py-2 text-xs font-medium bg-gradient-to-r from-blue-500 to-blue-600 text-white hover:from-blue-600 hover:to-blue-700 rounded-lg transition shadow-sm">
              <svg class="w-4 h-4 inline-block mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
              </svg>
              Show All Articles
            </button>
          </div>
          
          <div id="tagsList" class="space-y-1">
            {% for tag in all_tags %}
              <div class="tag-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="filterByTag('{{ tag }}', this)" data-tag="{{ tag }}">
                <span class="inline-block w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
                {{ tag }}
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
              <thead class="gradient-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;">
                <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;">
                  <th class="px-4 py-3 text-left text-xs font-semibold" style="color: white !important; background: transparent !important; width: 60px;">#</th>
                  <th class="px-4 py-3 text-left text-xs font-semibold" style="color: white !important; background: transparent !important;">Title</th>
                  <th class="px-4 py-3 text-left text-xs font-semibold" style="color: white !important; background: transparent !important;">Summary</th>
                  <th class="px-4 py-3 text-left text-xs font-semibold whitespace-nowrap" style="color: white !important; background: transparent !important;">Date</th>
                  <th class="px-4 py-3 text-left text-xs font-semibold" style="color: white !important; background: transparent !important;">Tags</th>
                  <th class="px-4 py-3 text-left text-xs font-semibold" style="color: white !important; background: transparent !important;">Categories</th>
                </tr>
              </thead>
              <tbody id="articlesTable" class="divide-y divide-gray-200">
                {% for post in sorted_posts %}
                  <tr class="article-row" 
                      data-tags="{% if post.tags %}{% for tag in post.tags %}{{ tag }}{% unless forloop.last %}|{% endunless %}{% endfor %}{% endif %}{% if post.tag and post.tags.size > 0 %}|{% endif %}{% if post.tag %}{{ post.tag }}{% endif %}"
                      data-categories="{% if post.categories %}{% for category in post.categories %}{{ category }}{% unless forloop.last %}|{% endunless %}{% endfor %}{% endif %}">
                    <td class="px-4 py-3 text-xs text-gray-600 font-medium">{{ forloop.index }}</td>
                    <td class="px-4 py-3">
                      <a href="{{ post.url | relative_url }}" class="text-blue-600 hover:text-blue-800 font-medium text-xs hover:underline">
                        {{ post.title | escape }}
                      </a>
                    </td>
                    <td class="px-4 py-3">
                      <div class="text-xs text-gray-600 line-clamp-2">
                        {% if post.summary %}
                          {{ post.summary }}
                        {% elsif post.excerpt %}
                          {{ post.excerpt | strip_html | truncatewords: 20 }}
                        {% else %}
                          <span class="text-gray-400 italic">No summary available</span>
                        {% endif %}
                      </div>
                    </td>
                    <td class="px-4 py-3 text-xs text-gray-600 whitespace-nowrap">
                      {{ post.date | date: "%b %-d, %Y" }}
                    </td>
                    <td class="px-4 py-3">
                      <div class="flex flex-wrap gap-1">
                        {% if post.tags %}
                          {% for tag in post.tags %}
                            <span class="inline-block px-2 py-0.5 text-xs font-medium rounded-full bg-blue-100 text-blue-700 hover:bg-blue-200 transition">
                              {{ tag }}
                            </span>
                          {% endfor %}
                        {% endif %}
                        {% if post.tag %}
                          <span class="inline-block px-2 py-0.5 text-xs font-medium rounded-full bg-blue-100 text-blue-700 hover:bg-blue-200 transition">
                            {{ post.tag }}
                          </span>
                        {% endif %}
                      </div>
                    </td>
                    <td class="px-4 py-3">
                      <div class="flex flex-wrap gap-1">
                        {% if post.categories %}
                          {% for category in post.categories %}
                            <span class="inline-block px-2 py-0.5 text-xs font-medium rounded-full bg-purple-100 text-purple-700 hover:bg-purple-200 transition">
                              {{ category }}
                            </span>
                          {% endfor %}
                        {% endif %}
                      </div>
                    </td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      <!-- Right Panel - Categories -->
      <div class="w-72 flex-shrink-0">
        <div class="side-panel bg-white rounded-xl shadow-lg p-4 border border-gray-100">
          <h3 class="text-base font-bold text-gray-800 mb-3 flex items-center gap-2 pb-2 border-b border-gray-200">
            <svg class="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
            </svg>
            <span>Categories</span>
            <span class="ml-auto text-sm font-normal text-gray-500">({{ all_categories.size }})</span>
          </h3>
          
          <div class="space-y-1">
            <!-- AI Trends -->
            <div>
              <div class="category-parent category-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="toggleCategoryExpand(this, 'ai-trends')">
                <span class="expand-icon">▶</span>
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mx-2"></span>
                AI Trends
              </div>
              <div class="subcategory-list" data-parent="ai-trends">
                {% for category in all_categories %}
                  {% if category contains "ai-trend" or category contains "ai_trend" %}
                    <div class="subcategory-item category-item px-3 py-2 rounded-lg" onclick="filterByCategory('{{ category }}', this)" data-category="{{ category }}">
                      <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                      {{ category }}
                    </div>
                  {% endif %}
                {% endfor %}
              </div>
            </div>
            
            <!-- Apps -->
            <div>
              <div class="category-parent category-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="toggleCategoryExpand(this, 'apps')">
                <span class="expand-icon">▶</span>
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mx-2"></span>
                Apps
              </div>
              <div class="subcategory-list" data-parent="apps">
                {% for category in all_categories %}
                  {% if category contains "app" %}
                    <div class="subcategory-item category-item px-3 py-2 rounded-lg" onclick="filterByCategory('{{ category }}', this)" data-category="{{ category }}">
                      <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                      {{ category }}
                    </div>
                  {% endif %}
                {% endfor %}
              </div>
            </div>
            
            <!-- Books -->
            <div>
              <div class="category-parent category-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="toggleCategoryExpand(this, 'books')">
                <span class="expand-icon">▶</span>
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mx-2"></span>
                Books
              </div>
              <div class="subcategory-list" data-parent="books">
                {% for category in all_categories %}
                  {% if category contains "book" or category contains "ai" and category contains "mysticism" or category contains "psych" or category contains "writing" %}
                    <div class="subcategory-item category-item px-3 py-2 rounded-lg" onclick="filterByCategory('{{ category }}', this)" data-category="{{ category }}">
                      <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                      {{ category }}
                    </div>
                  {% endif %}
                {% endfor %}
              </div>
            </div>
            
            <!-- Concepts -->
            <div>
              <div class="category-parent category-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="toggleCategoryExpand(this, 'concepts')">
                <span class="expand-icon">▶</span>
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mx-2"></span>
                Concepts
              </div>
              <div class="subcategory-list" data-parent="concepts">
                {% for category in all_categories %}
                  {% if category contains "genai" or category contains "agentic" or category contains "langraph" or category contains "pytorch" or category contains "fastai" or category contains "rag" or category contains "llm" or category contains "prompt" or category contains "reinforcement" or category contains "unsupervised" or category contains "android" or category contains "aws" or category contains "blockchain" or category contains "math" or category contains "physics" or category contains "english" or category contains "markdown" or category contains "music" or category contains "emotion" or category contains "humanism" or category contains "positive-psychology" or category contains "purposeful" or category contains "nptel" or category contains "coursework" or category contains "flight-dispatch" or category contains "weather" or category contains "windsurf" %}
                    <div class="subcategory-item category-item px-3 py-2 rounded-lg" onclick="filterByCategory('{{ category }}', this)" data-category="{{ category }}">
                      <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                      {{ category }}
                    </div>
                  {% endif %}
                {% endfor %}
              </div>
            </div>
            
            <!-- Spiritual/Philosophical -->
            <div>
              <div class="category-parent category-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="toggleCategoryExpand(this, 'spiritual')">
                <span class="expand-icon">▶</span>
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mx-2"></span>
                Spiritual/Philosophical
              </div>
              <div class="subcategory-list" data-parent="spiritual">
                {% for category in all_categories %}
                  {% if category contains "gita" or category contains "ramayana" or category contains "shloka" or category contains "panchatantra" or category contains "prem-rawat" or category contains "quote" or category contains "poem" or category contains "thought" %}
                    <div class="subcategory-item category-item px-3 py-2 rounded-lg" onclick="filterByCategory('{{ category }}', this)" data-category="{{ category }}">
                      <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                      {{ category }}
                    </div>
                  {% endif %}
                {% endfor %}
              </div>
            </div>
            
            <!-- Science -->
            <div>
              <div class="category-parent category-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="toggleCategoryExpand(this, 'science')">
                <span class="expand-icon">▶</span>
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mx-2"></span>
                Science
              </div>
              <div class="subcategory-list" data-parent="science">
                {% for category in all_categories %}
                  {% if category contains "science" %}
                    <div class="subcategory-item category-item px-3 py-2 rounded-lg" onclick="filterByCategory('{{ category }}', this)" data-category="{{ category }}">
                      <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                      {{ category }}
                    </div>
                  {% endif %}
                {% endfor %}
              </div>
            </div>
            
            <!-- Talks -->
            <div>
              <div class="category-parent category-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="toggleCategoryExpand(this, 'talks')">
                <span class="expand-icon">▶</span>
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mx-2"></span>
                Talks
              </div>
              <div class="subcategory-list" data-parent="talks">
                {% for category in all_categories %}
                  {% if category contains "talk" or category contains "seminar" or category contains "fsgai" %}
                    <div class="subcategory-item category-item px-3 py-2 rounded-lg" onclick="filterByCategory('{{ category }}', this)" data-category="{{ category }}">
                      <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                      {{ category }}
                    </div>
                  {% endif %}
                {% endfor %}
              </div>
            </div>
            
            <!-- Travels -->
            <div>
              <div class="category-parent category-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="toggleCategoryExpand(this, 'travels')">
                <span class="expand-icon">▶</span>
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mx-2"></span>
                Travels
              </div>
              <div class="subcategory-list" data-parent="travels">
                {% for category in all_categories %}
                  {% if category contains "travel" %}
                    <div class="subcategory-item category-item px-3 py-2 rounded-lg" onclick="filterByCategory('{{ category }}', this)" data-category="{{ category }}">
                      <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                      {{ category }}
                    </div>
                  {% endif %}
                {% endfor %}
              </div>
            </div>
            
            <!-- Songs -->
            <div>
              <div class="category-parent category-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="toggleCategoryExpand(this, 'songs')">
                <span class="expand-icon">▶</span>
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mx-2"></span>
                Songs
              </div>
              <div class="subcategory-list" data-parent="songs">
                {% for category in all_categories %}
                  {% if category contains "song" %}
                    <div class="subcategory-item category-item px-3 py-2 rounded-lg" onclick="filterByCategory('{{ category }}', this)" data-category="{{ category }}">
                      <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                      {{ category }}
                    </div>
                  {% endif %}
                {% endfor %}
              </div>
            </div>
          </div>
        </div>
      </div>
      
    </div>
  </div>
</div>

<script>
  let activeFilter = { type: null, value: null };
  
  function toggleCategoryExpand(element, parentId) {
    const parent = element;
    const subcategoryList = document.querySelector(`.subcategory-list[data-parent="${parentId}"]`);
    
    if (parent.classList.contains('expanded')) {
      parent.classList.remove('expanded');
      subcategoryList.classList.remove('expanded');
    } else {
      parent.classList.add('expanded');
      subcategoryList.classList.add('expanded');
    }
    
    // Stop event propagation to prevent filtering when expanding/collapsing
    event.stopPropagation();
  }
  
  function updateFilteredCount() {
    const visibleRows = document.querySelectorAll('.article-row:not(.hidden-row)').length;
    document.getElementById('filteredCountTop').textContent = visibleRows;
  }
  
  function filterByTag(tag, element) {
    const tagItems = document.querySelectorAll('.tag-item');
    const categoryItems = document.querySelectorAll('.category-item');
    
    tagItems.forEach(item => item.classList.remove('active'));
    categoryItems.forEach(item => item.classList.remove('active'));
    
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
  
  function filterByCategory(category, element) {
    const tagItems = document.querySelectorAll('.tag-item');
    const categoryItems = document.querySelectorAll('.category-item');
    
    tagItems.forEach(item => item.classList.remove('active'));
    categoryItems.forEach(item => item.classList.remove('active'));
    
    element.classList.add('active');
    activeFilter = { type: 'category', value: category };
    
    const rows = document.querySelectorAll('.article-row');
    rows.forEach(row => {
      const categories = row.dataset.categories ? row.dataset.categories.split('|') : [];
      if (categories.includes(category)) {
        row.classList.remove('hidden-row');
      } else {
        row.classList.add('hidden-row');
      }
    });
    
    updateFilteredCount();
  }
  
  function clearFilters() {
    const tagItems = document.querySelectorAll('.tag-item');
    const categoryItems = document.querySelectorAll('.category-item');
    const categoryParents = document.querySelectorAll('.category-parent');
    const rows = document.querySelectorAll('.article-row');
    
    tagItems.forEach(item => item.classList.remove('active'));
    categoryItems.forEach(item => item.classList.remove('active'));
    categoryParents.forEach(parent => parent.classList.remove('active'));
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
