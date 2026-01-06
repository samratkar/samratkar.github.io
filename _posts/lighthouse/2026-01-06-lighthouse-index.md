---
date: 2026-01-06 
layout: default
title: Lighthouse - Wisdom & Reflection
category: lighthouse
subcategory: index
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
              <div class="text-2xl font-bold text-gray-800">2</div>
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
              <div class="text-2xl font-bold text-gray-800" id="tagCount">6</div>
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
              <div class="text-2xl font-bold text-gray-800" id="topicCount">2</div>
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
              <div class="text-2xl font-bold text-gray-800" id="filteredCount">2</div>
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
              <div class="tag-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="filterByTag('freedom', this)" data-tag="freedom">
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                Freedom
              </div>
              <div class="tag-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="filterByTag('courage', this)" data-tag="courage">
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                Courage
              </div>
              <div class="tag-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="filterByTag('meaning', this)" data-tag="meaning">
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                Meaning
              </div>
              <div class="tag-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="filterByTag('authenticity', this)" data-tag="authenticity">
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                Authenticity
              </div>
              <div class="tag-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="filterByTag('philosophy', this)" data-tag="philosophy">
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                Philosophy
              </div>
              <div class="tag-item px-3 py-2.5 rounded-lg text-sm font-medium" onclick="filterByTag('psychology', this)" data-tag="psychology">
                <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                Psychology
              </div>
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
                  <tr class="article-row" 
                      data-tags="freedom|courage|authenticity|psychology"
                      data-topic="Adlerian Psychology"
                      data-subtopics="The Past|Suffering|Courage to Be Disliked|Separation of Tasks|Comparison|Contribution|Happiness|Way of Being">
                    <td class="px-4 py-3 text-sm text-gray-600 font-medium">1</td>
                    <td class="px-4 py-3">
                      <a href="#unborrowed-ground" class="text-purple-600 hover:text-purple-800 font-medium text-sm hover:underline">
                        The Unborrowed Ground
                      </a>
                    </td>
                    <td class="px-4 py-3">
                      <div class="text-sm text-gray-600 line-clamp-2">
                        Exploring Adlerian psychology through poetry — the past is not a prison, suffering is relational, and freedom comes from the courage to be disliked. Discover the power of contribution over recognition.
                      </div>
                    </td>
                    <td class="px-4 py-3">
                      <div class="flex flex-wrap gap-1">
                        <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-700">freedom</span>
                        <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-700">courage</span>
                        <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-700">authenticity</span>
                        <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-700">psychology</span>
                      </div>
                    </td>
                  </tr>
                  <tr class="article-row" 
                      data-tags="meaning|freedom|philosophy|courage"
                      data-topic="Tagore's Philosophy"
                      data-subtopics="Awakening of Meaning|Freedom from Past|Relational Vastness|Divine Immanence|Comradeship|Embrace of Life|Divine in Humanity|Universe Made Human">
                    <td class="px-4 py-3 text-sm text-gray-600 font-medium">2</td>
                    <td class="px-4 py-3">
                      <a href="#the-stance" class="text-purple-600 hover:text-purple-800 font-medium text-sm hover:underline">
                        The Stance
                      </a>
                    </td>
                    <td class="px-4 py-3">
                      <div class="text-sm text-gray-600 line-clamp-2">
                        "The truth of the Universe is human truth." Journey through Tagore's philosophy — the awakening of meaning, freedom from the tyranny of the past, and the divine woven into humanity. Faith that sings before dawn.
                      </div>
                    </td>
                    <td class="px-4 py-3">
                      <div class="flex flex-wrap gap-1">
                        <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-700">meaning</span>
                        <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-700">freedom</span>
                        <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-700">philosophy</span>
                        <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-700">courage</span>
                      </div>
                    </td>
                  </tr>
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
              <div class="topic-section">
                <div class="topic-item topic-header px-3 py-2.5 rounded-lg text-sm flex items-center justify-between" onclick="toggleTopic('adlerian', this)" data-topic="Adlerian Psychology">
                  <div>
                    <span class="inline-block w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                    <span>Adlerian Psychology</span>
                  </div>
                  <svg class="w-4 h-4 transform transition-transform topic-chevron" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
                <div id="adlerian-subtopics" class="subtopics-container">
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('The Past', this)" data-subtopic="The Past">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    1. The Past Is Not a Prison
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Suffering', this)" data-subtopic="Suffering">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    2. All Suffering Is Relational
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Courage to Be Disliked', this)" data-subtopic="Courage to Be Disliked">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    3. The Courage to Be Disliked
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Separation of Tasks', this)" data-subtopic="Separation of Tasks">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    4. The Separation of Tasks
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Comparison', this)" data-subtopic="Comparison">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    5. Beyond Comparison
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Contribution', this)" data-subtopic="Contribution">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    6. Contribution Over Recognition
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Happiness', this)" data-subtopic="Happiness">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    7. Happiness Is a Choice
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Way of Being', this)" data-subtopic="Way of Being">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    8. Courage as a Way of Being
                  </div>
                </div>
              </div>
              
              <div class="topic-section">
                <div class="topic-item topic-header px-3 py-2.5 rounded-lg text-sm flex items-center justify-between" onclick="toggleTopic('tagore', this)" data-topic="Tagore's Philosophy">
                  <div>
                    <span class="inline-block w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                    <span>Tagore's Philosophy</span>
                  </div>
                  <svg class="w-4 h-4 transform transition-transform topic-chevron" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
                <div id="tagore-subtopics" class="subtopics-container">
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Awakening of Meaning', this)" data-subtopic="Awakening of Meaning">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    1. The Awakening of Meaning
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Freedom from Past', this)" data-subtopic="Freedom from Past">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    2. Freedom from Tyranny of Past
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Relational Vastness', this)" data-subtopic="Relational Vastness">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    3. The Relational Vastness & Awe
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Divine Immanence', this)" data-subtopic="Divine Immanence">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    4. Immanence of the Divine
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Comradeship', this)" data-subtopic="Comradeship">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    5. Comradeship of All Beings
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Embrace of Life', this)" data-subtopic="Embrace of Life">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    6. Courageous Embrace of Life
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Divine in Humanity', this)" data-subtopic="Divine in Humanity">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    7. The Divine Woven into Humanity
                  </div>
                  <div class="subtopic-item px-3 py-2 rounded-lg text-xs" onclick="filterBySubtopic('Universe Made Human', this)" data-subtopic="Universe Made Human">
                    <span class="inline-block w-1.5 h-1.5 bg-purple-300 rounded-full mr-2"></span>
                    8. The Universe Made Human
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
      </div>
    </div>
  </div>
  
  <!-- Full Articles Section -->
  <div class="full-width-container bg-gradient-to-b from-white to-purple-50 py-12 mt-12">
    <div class="max-w-5xl mx-auto px-6">
      <div class="text-center mb-12">
        <h2 class="text-4xl font-bold text-gray-800 mb-4" style="font-family: 'Crimson Text', serif;">Full Articles</h2>
        <div class="w-32 h-1 bg-gradient-to-r from-purple-400 via-purple-500 to-purple-600 mx-auto rounded-full"></div>
        <p class="mt-4 text-gray-600 text-lg">Dive deep into each reflection below</p>
      </div>
      
      <div id="unborrowed-ground" class="bg-white rounded-2xl shadow-xl p-8 mb-8 border-l-8 border-purple-500 hover:shadow-2xl transition-shadow">
        <h2 class="text-4xl font-bold text-purple-900 mb-6" style="font-family: 'Crimson Text', serif;">The Unborrowed Ground</h2>
        <div class="prose prose-lg max-w-none">
          {% include lighthouse/001-unborrowed-ground.md %}
        </div>
      </div>
      
      <div id="the-stance" class="bg-white rounded-2xl shadow-xl p-8 mb-8 border-l-8 border-purple-400 hover:shadow-2xl transition-shadow">
        <h2 class="text-4xl font-bold text-purple-900 mb-6" style="font-family: 'Crimson Text', serif;">The Stance</h2>
        <div class="prose prose-lg max-w-none">
          {% include lighthouse/002-the-stance.md %}
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
