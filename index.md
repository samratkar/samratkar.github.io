---
layout: default
title: Home - All Articles
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
    background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%) !important;
  }
  
  .gradient-bg h1,
  .gradient-bg p {
    color: white !important;
  }
  
  .gradient-bg p.subtitle {
    color: rgba(233, 213, 255, 1) !important;
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
  
  table thead.gradient-header {
    background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%) !important;
  }
  
  .glass-effect {
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(196, 181, 253, 0.4);
    box-shadow: 0 4px 16px rgba(91, 33, 182, 0.1);
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
  
  .tag-item, .category-item {
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .tag-item:hover, .category-item:hover {
    background-color: #f3e8ff;
    transform: translateX(4px);
  }
  
  .tag-item.active, .category-item.active {
    background-color: #9b87f5;
    color: white;
    font-weight: 600;
  }
  
  .subcategory-item {
    cursor: pointer;
    transition: all 0.2s ease;
    margin-left: 1rem;
    font-size: 0.85rem;
  }
  
  .subcategory-item:hover {
    background-color: #f3e8ff;
    transform: translateX(4px);
  }
  
  .subcategory-item.active {
    background-color: #c4b5fd;
    color: white;
    font-weight: 600;
  }
  
  /* Style for @mentions - lavender theme color */
  .at-mention {
    color: #5b21b6;
    font-weight: 500;
  }
  
  .category-header {
    cursor: pointer;
    font-weight: 600;
  }
  
  .category-section {
    margin-bottom: 0.75rem;
  }
  
  .article-row {
    transition: background-color 0.2s ease;
    background: white;
  }
  
  .article-row:hover {
    background-color: #faf5ff;
  }
  
  .hidden-row {
    display: none;
  }
  
  /* Ensure three-column layout stays horizontal */
  .flex.gap-6 {
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    align-items: flex-start;
  }
  
  /* Adjust panel widths for better table visibility */
  .w-72.flex-shrink-0 {
    width: 15rem !important;
  }
  
  /* Table container with vertical scroll */
  .flex-1.min-w-0 > div {
    max-height: calc(100vh - 12rem);
    overflow-y: auto;
  }
  
  .flex-1.min-w-0 > div::-webkit-scrollbar {
    width: 8px;
  }
  
  .flex-1.min-w-0 > div::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
  }
  
  .flex-1.min-w-0 > div::-webkit-scrollbar-thumb {
    background: #9b87f5;
    border-radius: 10px;
  }
  
  .flex-1.min-w-0 > div::-webkit-scrollbar-thumb:hover {
    background: #8b5cf6;
  }
  
  /* Table column widths */
  table th:nth-child(1) { width: 50px; }
  table th:nth-child(2) { width: 25%; }
  table th:nth-child(3) { width: 30%; }
  table th:nth-child(4) { width: 100px; }
  table th:nth-child(5) { width: 15%; }
  table th:nth-child(6) { width: 15%; }
  
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
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.85) 0%, rgba(243, 232, 255, 0.6) 100%) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    box-shadow: 
      6px 6px 12px rgba(155, 135, 245, 0.12),
      -3px -3px 8px rgba(255, 255, 255, 0.7),
      inset 1px 1px 2px rgba(255, 255, 255, 0.8),
      inset -1px -1px 2px rgba(155, 135, 245, 0.08);
    border: none;
    transition: all 0.3s ease;
  }
  
  .stat-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 
      8px 8px 16px rgba(155, 135, 245, 0.18),
      -4px -4px 10px rgba(255, 255, 255, 0.8),
      inset 1px 1px 2px rgba(255, 255, 255, 0.9),
      inset -1px -1px 2px rgba(155, 135, 245, 0.1);
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.92) 0%, rgba(243, 232, 255, 0.75) 100%) !important;
  }
  
  .stat-card .flex {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 0.75rem;
  }
  
  .stat-icon {
    width: 36px;
    height: 36px;
    min-width: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0;
    flex-shrink: 0;
  }
  
  .stat-icon svg {
    width: 20px;
    height: 20px;
  }
  
  /* Hide default theme header */
  .site-header {
    display: none !important;
  }
  
  header.site-header {
    display: none !important;
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
  }
</style>

<div class="bg-gray-50 min-h-screen">
  <!-- Header - Full Width -->
  <header class="full-width-header shadow-lg" style="background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%) !important;">
    <div class="max-w-screen-2xl mx-auto px-6 py-6">
      <div class="flex items-center justify-between flex-wrap gap-6">
        <div class="flex-1">
          <h1 class="text-3xl font-bold" style="color: white !important; font-family: 'Crimson Text', serif; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">📚 Samrat Kar | exploring & experimenting 📚</h1>
          <p class="subtitle mt-2 text-sm" style="color: #e9d5ff !important;">Article Index - All my writings and explorations</p>
        </div>
        <nav class="nav-menu">
          <a href="/" class="nav-link" style="color: white !important;">🏠 Home</a>
          <a href="/about/" class="nav-link" style="color: white !important;">👤 About</a>
          <a href="/books/" class="nav-link" style="color: white !important;">📚 Books</a>
        </nav>
      </div>
    </div>
  </header>

  <!-- Collect all tags, categories, and subcategories from all posts FIRST -->
  {% assign all_tags = "" | split: "" %}
  {% assign all_categories = "" | split: "" %}
  {% assign all_subcategories = "" | split: "" %}
  {% comment %} Create a map of category to subcategories {% endcomment %}
  {% assign category_subcategories = "" %}
  {% for post in site.posts %}
    <!-- Collect from both 'tags' and 'tag' fields -->
    {% if post.tags %}
      {% for tag in post.tags %}
        {% assign tag_str = tag | strip %}
        {% unless all_tags contains tag_str %}
          {% assign all_tags = all_tags | push: tag_str %}
        {% endunless %}
      {% endfor %}
    {% endif %}
    {% if post.tag %}
      {% if post.tag.first %}
        <!-- If post.tag is an array, iterate through it -->
        {% for tag in post.tag %}
          {% assign tag_str = tag | strip %}
          {% unless all_tags contains tag_str %}
            {% assign all_tags = all_tags | push: tag_str %}
          {% endunless %}
        {% endfor %}
      {% else %}
        <!-- If post.tag is a single string -->
        {% assign tag_value = post.tag | strip %}
        {% unless all_tags contains tag_value %}
          {% assign all_tags = all_tags | push: tag_value %}
        {% endunless %}
      {% endif %}
    {% endif %}
    <!-- Collect categories -->
    {% if post.categories %}
      {% for category in post.categories %}
        {% assign cat_str = category | strip %}
        {% unless all_categories contains cat_str %}
          {% assign all_categories = all_categories | push: cat_str %}
        {% endunless %}
      {% endfor %}
    {% endif %}
    {% if post.category %}
      {% assign cat_value = post.category | strip %}
      {% unless all_categories contains cat_value %}
        {% assign all_categories = all_categories | push: cat_value %}
      {% endunless %}
      {% comment %} Build category-subcategory mapping {% endcomment %}
      {% if post.subcategory %}
        {% assign mapping_key = cat_value | append: ":::" | append: post.subcategory %}
        {% unless category_subcategories contains mapping_key %}
          {% assign category_subcategories = category_subcategories | append: mapping_key | append: "|||" %}
        {% endunless %}
      {% endif %}
    {% endif %}
    <!-- Collect subcategories -->
    {% if post.subcategories %}
      {% for subcategory in post.subcategories %}
        {% assign subcat_str = subcategory | strip %}
        {% unless all_subcategories contains subcat_str %}
          {% assign all_subcategories = all_subcategories | push: subcat_str %}
        {% endunless %}
      {% endfor %}
    {% endif %}
    {% if post.subcategory %}
      {% assign subcat_value = post.subcategory | strip %}
      {% unless all_subcategories contains subcat_value %}
        {% assign all_subcategories = all_subcategories | push: subcat_value %}
      {% endunless %}
    {% endif %}
  {% endfor %}
  {% assign all_tags = all_tags | sort %}
  {% assign all_categories = all_categories | sort %}
  {% assign all_subcategories = all_subcategories | sort %}

  <!-- Statistics Section -->
  <div class="full-width-container border-b border-gray-200 shadow-sm" style="background: linear-gradient(135deg, rgba(155, 135, 245, 0.1) 0%, rgba(196, 181, 253, 0.15) 100%);">
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
      
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="stat-card">
          <div class="flex items-center gap-3">
            <div class="stat-icon" style="background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%);">
              <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
            </div>
            <div class="flex items-baseline gap-2">
              <span class="text-2xl font-bold text-gray-800">{{ total_posts }}</span>
              <span class="text-sm text-gray-500">Articles</span>
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
            <div class="flex items-baseline gap-2">
              <span class="text-2xl font-bold text-gray-800">{{ all_tags.size }}</span>
              <span class="text-sm text-gray-500">Tags</span>
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
            <div class="flex items-baseline gap-2">
              <span class="text-2xl font-bold text-gray-800">{{ all_categories.size }}</span>
              <span class="text-sm text-gray-500">Categories</span>
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
            <div class="flex items-baseline gap-2">
              <span class="text-2xl font-bold text-gray-800" id="filteredCountTop">{{ total_posts }}</span>
              <span class="text-sm text-gray-500">Filtered</span>
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
      <div class="w-72 flex-shrink-0" style="width: 15rem;">
        <div class="side-panel rounded-xl shadow-lg p-4" style="background: white; border: 1px solid rgba(196, 181, 253, 0.3);">
          <h3 class="text-base font-bold text-purple-900 mb-3 flex items-center gap-2 pb-2 border-b border-purple-200">
            <svg class="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"></path>
            </svg>
            <span>Tags</span>
            <span class="ml-auto text-sm font-normal text-purple-600">({{ all_tags.size }})</span>
          </h3>
          
          <div class="mb-4">
            <input 
              type="text" 
              id="tagSearch" 
              placeholder="Search tags..." 
              class="w-full px-3 py-2 text-sm rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              style="background: rgba(255, 255, 255, 0.7); border: 1px solid rgba(196, 181, 253, 0.5);"
              onkeyup="filterTagsList()"
            >
          </div>
          
          <div class="mb-3">
            <button onclick="clearFilters()" class="w-full px-3 py-2 text-xs font-medium bg-gradient-to-r from-purple-500 to-purple-600 text-white hover:from-purple-600 hover:to-purple-700 rounded-lg transition shadow-sm">>
              <svg class="w-4 h-4 inline-block mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
              </svg>
              Show All Articles
            </button>
          </div>
          
          <div id="tagsList" class="space-y-1">
            {% for tag in all_tags %}
              <div class="tag-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="filterByTag('{{ tag }}', this)" data-tag="{{ tag }}">
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                {{ tag }}
              </div>
            {% endfor %}
          </div>
        </div>
      </div>
      
      <!-- Center - Articles Table -->
      <div class="flex-1 min-w-0">
        <div class="rounded-xl shadow-lg overflow-hidden" style="background: white; border: 1px solid rgba(196, 181, 253, 0.5);">
          <div class="overflow-x-auto">
            <table class="w-full">
              <thead class="gradient-header" style="background: linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%) !important;">
                <tr style="background: transparent !important;">
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
                      data-tags="{% if post.tags %}{% for tag in post.tags %}{{ tag }}{% unless forloop.last %}|{% endunless %}{% endfor %}{% endif %}{% if post.tag %}{% if post.tag.first %}{% for tag in post.tag %}{{ tag }}{% unless forloop.last %}|{% endunless %}{% endfor %}{% else %}{{ post.tag }}{% endif %}{% endif %}"
                      data-categories="{% for category in post.categories %}{{ category }}{% unless forloop.last %}|{% endunless %}{% endfor %}"
                      data-subcategories="{% if post.subcategory %}{{ post.subcategory }}{% endif %}{% if post.subcategories %}{% for subcategory in post.subcategories %}{{ subcategory }}{% unless forloop.last %}|{% endunless %}{% endfor %}{% endif %}">
                    <td class="px-4 py-3 text-xs text-gray-600 font-medium">{{ forloop.index }}</td>
                    <td class="px-4 py-3">
                      <a href="{{ post.url | relative_url }}" class="text-purple-600 hover:text-purple-800 font-medium text-xs hover:underline">
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
                        {% assign displayed_tags = "" | split: "" %}
                        {% if post.tags %}
                          {% for tag in post.tags %}
                            {% unless displayed_tags contains tag %}
                              <span class="inline-block px-2 py-0.5 text-xs font-medium rounded-full bg-purple-100 text-purple-700 hover:bg-purple-200 transition">
                                {{ tag }}
                              </span>
                              {% assign displayed_tags = displayed_tags | push: tag %}
                            {% endunless %}
                          {% endfor %}
                        {% endif %}
                        {% if post.tag %}
                          {% if post.tag.first %}
                            {% for tag in post.tag %}
                              {% unless displayed_tags contains tag %}
                                <span class="inline-block px-2 py-0.5 text-xs font-medium rounded-full bg-purple-100 text-purple-700 hover:bg-purple-200 transition">
                                  {{ tag }}
                                </span>
                                {% assign displayed_tags = displayed_tags | push: tag %}
                              {% endunless %}
                            {% endfor %}
                          {% else %}
                            {% unless displayed_tags contains post.tag %}
                              <span class="inline-block px-2 py-0.5 text-xs font-medium rounded-full bg-purple-100 text-purple-700 hover:bg-purple-200 transition">
                                {{ post.tag }}
                              </span>
                            {% endunless %}
                          {% endif %}
                        {% endif %}
                      </div>
                    </td>
                    <td class="px-4 py-3">
                      <div class="flex flex-wrap gap-1">
                        {% for category in post.categories %}
                          <span class="inline-block px-2 py-0.5 text-xs font-medium rounded-full bg-purple-100 text-purple-700 hover:bg-purple-200 transition">
                            {{ category }}
                          </span>
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
      
      <!-- Right Panel - Categories & Sub-Categories -->
      <div class="w-72 flex-shrink-0" style="width: 15rem;">
        <div class="side-panel rounded-xl shadow-lg p-4" style="background: white; border: 1px solid rgba(196, 181, 253, 0.3);">
          <h3 class="text-base font-bold text-purple-900 mb-3 flex items-center gap-2 pb-2 border-b border-purple-200">
            <svg class="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
            </svg>
            <span>Categories</span>
            <span class="ml-auto text-sm font-normal text-purple-600">({{ all_categories.size }})</span>
          </h3>
          
          <div class="space-y-1">
            {% for category in all_categories %}
              <div class="category-section">
                <div class="category-item category-header px-3 py-2.5 rounded-lg text-sm" onclick="filterByCategory('{{ category }}', this)" data-category="{{ category }}">
                  <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                  {{ category }}
                </div>
                {% comment %} Find and display subcategories for this category {% endcomment %}
                {% assign category_key = category | append: ":::" %}
                {% if category_subcategories contains category_key %}
                  <div class="ml-4 mt-1 space-y-1">
                    {% assign mappings = category_subcategories | split: "|||" %}
                    {% for mapping in mappings %}
                      {% if mapping contains category_key %}
                        {% assign parts = mapping | split: ":::" %}
                        {% if parts[0] == category %}
                          <a href="javascript:void(0)" class="subcategory-item block px-3 py-1.5 rounded-lg text-sm hover:no-underline" onclick="filterBySubcategory('{{ parts[1] }}', this); return false;" data-subcategory="{{ parts[1] }}">
                            <span class="inline-block w-1.5 h-1.5 bg-purple-400 rounded-full mr-2"></span>
                            <span class="text-purple-700">{{ parts[1] }}</span>
                          </a>
                        {% endif %}
                      {% endif %}
                    {% endfor %}
                  </div>
                {% endif %}
              </div>
            {% endfor %}
          </div>
        </div>
      </div>
      
    </div>
  </div>
</div>

<script>
  let activeFilter = { type: null, value: null };
  
  function updateFilteredCount() {
    const visibleRows = document.querySelectorAll('.article-row:not(.hidden-row)').length;
    document.getElementById('filteredCountTop').textContent = visibleRows;
  }
  
  function filterByTag(tag, element) {
    const tagItems = document.querySelectorAll('.tag-item');
    const categoryItems = document.querySelectorAll('.category-item');
    const subcategoryItems = document.querySelectorAll('.subcategory-item');
    
    tagItems.forEach(item => item.classList.remove('active'));
    categoryItems.forEach(item => item.classList.remove('active'));
    subcategoryItems.forEach(item => item.classList.remove('active'));
    
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
    const subcategoryItems = document.querySelectorAll('.subcategory-item');
    
    tagItems.forEach(item => item.classList.remove('active'));
    categoryItems.forEach(item => item.classList.remove('active'));
    subcategoryItems.forEach(item => item.classList.remove('active'));
    
    element.classList.add('active');
    activeFilter = { type: 'category', value: category };
    
    const rows = document.querySelectorAll('.article-row');
    rows.forEach(row => {
      const categories = row.dataset.categories ? row.dataset.categories.split('|').filter(c => c.trim()) : [];
      if (categories.includes(category)) {
        row.classList.remove('hidden-row');
      } else {
        row.classList.add('hidden-row');
      }
    });
    
    updateFilteredCount();
  }
  
  function filterBySubcategory(subcategory, element) {
    const tagItems = document.querySelectorAll('.tag-item');
    const categoryItems = document.querySelectorAll('.category-item');
    const subcategoryItems = document.querySelectorAll('.subcategory-item');
    
    tagItems.forEach(item => item.classList.remove('active'));
    categoryItems.forEach(item => item.classList.remove('active'));
    subcategoryItems.forEach(item => item.classList.remove('active'));
    
    element.classList.add('active');
    activeFilter = { type: 'subcategory', value: subcategory };
    
    const rows = document.querySelectorAll('.article-row');
    rows.forEach(row => {
      const subcategories = row.dataset.subcategories ? row.dataset.subcategories.split('|').filter(s => s.trim()) : [];
      if (subcategories.includes(subcategory)) {
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
    const subcategoryItems = document.querySelectorAll('.subcategory-item');
    const rows = document.querySelectorAll('.article-row');
    
    tagItems.forEach(item => item.classList.remove('active'));
    categoryItems.forEach(item => item.classList.remove('active'));
    subcategoryItems.forEach(item => item.classList.remove('active'));
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
  
  // Style words starting with @ in blue
  function styleAtMentions() {
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      null,
      false
    );
    
    const textNodes = [];
    while (walker.nextNode()) {
      if (walker.currentNode.nodeValue.match(/@[a-zA-Z_][a-zA-Z0-9_]*/)) {
        textNodes.push(walker.currentNode);
      }
    }
    
    textNodes.forEach(node => {
      const text = node.nodeValue;
      const regex = /@[a-zA-Z_][a-zA-Z0-9_]*/g;
      
      if (regex.test(text)) {
        const span = document.createElement('span');
        span.innerHTML = text.replace(/@[a-zA-Z_][a-zA-Z0-9_]*/g, '<span class="at-mention">$&</span>');
        node.parentNode.replaceChild(span, node);
      }
    });
  }
  
  // Run on page load
  document.addEventListener('DOMContentLoaded', styleAtMentions);
  
  // Re-apply stat-card styles after Tailwind loads
  function applyStatCardStyles() {
    document.querySelectorAll('.stat-card').forEach(card => {
      card.style.cssText = `
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.85) 0%, rgba(243, 232, 255, 0.6) 100%) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        box-shadow: 6px 6px 12px rgba(155, 135, 245, 0.12), -3px -3px 8px rgba(255, 255, 255, 0.7), inset 1px 1px 2px rgba(255, 255, 255, 0.8), inset -1px -1px 2px rgba(155, 135, 245, 0.08);
        border: none;
        transition: all 0.3s ease;
      `;
    });
    
    // Apply header gradient
    document.querySelectorAll('.full-width-header').forEach(header => {
      header.style.background = 'linear-gradient(135deg, #9b87f5 0%, #c4b5fd 100%)';
    });
  }
  
  // Apply immediately and after a short delay (for Tailwind)
  document.addEventListener('DOMContentLoaded', applyStatCardStyles);
  setTimeout(applyStatCardStyles, 100);
  setTimeout(applyStatCardStyles, 500);
</script>
