---
layout: default
title: About Me
permalink: /about/
---

<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
  body {
    font-family: 'Inter', sans-serif;
    max-width: 100% !important;
    margin: 0 !important;
  }
  
  .container, .wrapper {
    max-width: 100% !important;
  }
  
  .gradient-bg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }
  
  .profile-card {
    background: white;
    border-radius: 0.75rem;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
  }
  
  .profile-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
  }
  
  .section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #667eea;
    display: inline-block;
  }
  
  .badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    margin: 0.25rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
    transition: all 0.2s;
  }
  
  .badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }
  
  .badge-purple {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }
  
  .badge-blue {
    background: #dbeafe;
    color: #1e40af;
  }
  
  .badge-green {
    background: #d1fae5;
    color: #065f46;
  }
  
  .badge-pink {
    background: #fce7f3;
    color: #9f1239;
  }
  
  .timeline-item {
    position: relative;
    padding-left: 2rem;
    padding-bottom: 1.5rem;
    border-left: 2px solid #e5e7eb;
  }
  
  .timeline-item:last-child {
    border-left: none;
  }
  
  .timeline-dot {
    position: absolute;
    left: -0.5rem;
    top: 0;
    width: 1rem;
    height: 1rem;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }
  
  .stat-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 1rem;
    padding: 1.5rem;
    text-align: center;
  }
  
  .stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }
  
  .stat-label {
    font-size: 0.875rem;
    opacity: 0.9;
  }
  
  .social-link {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: white;
    border: 2px solid #e5e7eb;
    border-radius: 0.5rem;
    color: #667eea;
    text-decoration: none;
    transition: all 0.2s;
    font-weight: 500;
  }
  
  .social-link:hover {
    background: #667eea;
    color: white;
    border-color: #667eea;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
  }
  
  .profile-photo {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    border: 5px solid white;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    object-fit: cover;
  }
  
  .experience-card {
    background: white;
    border-radius: 0.75rem;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border-left: 3px solid #667eea;
    transition: all 0.3s ease;
  }
  
  .experience-card:hover {
    transform: translateX(8px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    border-left-width: 6px;
  }
  
  .experience-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 1rem;
  }
  
  .experience-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.3rem;
  }
  
  .experience-company {
    font-size: 0.9rem;
    color: #667eea;
    font-weight: 600;
    margin-bottom: 0.2rem;
  }
  
  .experience-duration {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 600;
  }
  
  .experience-department {
    color: #6b7280;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 0.75rem;
  }
  
  .achievement-list {
    list-style: none;
    padding: 0;
    margin: 1rem 0;
  }
  
  .achievement-list li {
    position: relative;
    padding-left: 2rem;
    margin-bottom: 0.75rem;
    color: #4b5563;
    line-height: 1.6;
    font-size: 0.875rem;
  }
  
  .achievement-list li:before {
    content: "→";
    position: absolute;
    left: 0;
    color: #667eea;
    font-weight: bold;
    font-size: 1.2rem;
  }
  
  .highlight-metric {
    background: #fef3c7;
    color: #92400e;
    padding: 0.15rem 0.5rem;
    border-radius: 0.25rem;
    font-weight: 600;
  }
  
  .export-resume-btn {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 9999px;
    font-weight: 600;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    z-index: 1000;
    border: none;
    font-size: 1rem;
  }
  
  .export-resume-btn:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.5);
  }
  
  .main-layout {
    display: flex;
    gap: 2rem;
    max-width: 80%;
    margin: 0 auto;
  }
  
  .main-content {
    flex: 1;
    min-width: 0;
  }
  
  .right-panel {
    width: 320px;
    flex-shrink: 0;
  }
  
  .right-panel-sticky {
    position: sticky;
    top: 2rem;
    max-height: calc(100vh - 4rem);
    overflow-y: auto;
  }
  
  .right-panel .profile-card {
    padding: 1.25rem;
    margin-bottom: 1rem;
    font-size: 0.875rem;
  }
  
  .right-panel .section-title {
    font-size: 1.1rem;
    margin-bottom: 0.75rem;
  }
  
  @media (max-width: 1024px) {
    .main-layout {
      flex-direction: column;
    }
    .right-panel {
      width: 100%;
    }
    .right-panel-sticky {
      position: relative;
      max-height: none;
    }
  }
  
  /* Print Styles for Resume Export */
  @media print {
    body {
      background: white !important;
      font-size: 10pt;
    }
    
    .export-resume-btn,
    .back-link,
    nav,
    footer,
    .stat-card,
    .social-link,
    .badge:not(.experience-duration) {
      display: none !important;
    }
    
    .gradient-bg {
      background: #667eea !important;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
      padding: 1rem !important;
    }
    
    .profile-photo {
      width: 120px;
      height: 120px;
    }
    
    h1 {
      font-size: 24pt !important;
    }
    
    h2.section-title {
      font-size: 16pt !important;
      page-break-after: avoid;
    }
    
    h3.experience-title {
      font-size: 14pt !important;
      page-break-after: avoid;
    }
    
    .experience-card {
      page-break-inside: avoid;
      margin-bottom: 1rem;
      box-shadow: none !important;
      border: 1px solid #e5e7eb;
    }
    
    .profile-card {
      page-break-inside: avoid;
      box-shadow: none !important;
      border: 1px solid #e5e7eb;
      margin-bottom: 1rem;
    }
    
    .highlight-metric {
      background: #fef3c7 !important;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }
    
    .experience-duration {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }
    
    a {
      color: #667eea !important;
      text-decoration: none;
    }
    
    .achievement-list li {
      margin-bottom: 0.5rem;
      line-height: 1.4;
    }
    
    @page {
      margin: 1cm;
    }
  }
</style>

<script>
  function exportResume() {
    window.print();
  }
</script>

<div class="bg-gray-50 min-h-screen">
  
  <!-- Back to Home Link -->
  <div class="px-8 pt-6">
    <a href="/" class="inline-flex items-center gap-2 text-gray-600 hover:text-purple-600 transition-colors duration-200">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
      </svg>
      <span class="font-medium">Back to Home</span>
    </a>
  </div>
  
  <!-- Export Resume Button -->
  <button class="export-resume-btn" onclick="exportResume()">
    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
    </svg>
    Export Resume
  </button>
  
  <!-- Hero Section -->
  <div class="gradient-bg text-white py-16">
    <div class="px-8">
      <div class="flex flex-col md:flex-row items-center gap-8 max-w-[80%] mx-auto">
        <img src="{{ site.data.about.profile.photo }}" alt="{{ site.data.about.profile.name }}" class="profile-photo">
        <div class="text-center md:text-left">
          <h1 class="text-2xl font-bold mb-2">{{ site.data.about.profile.name }}</h1>
          <p class="text-base text-blue-100 mb-2">{{ site.data.about.profile.title }}</p>
          <p class="text-sm opacity-90">{{ site.data.about.profile.organization }}</p>
          <div class="flex flex-wrap gap-3 mt-4 justify-center md:justify-start">
            {% for badge in site.data.about.profile.badges %}
            <span class="badge badge-purple">{{ badge }}</span>
            {% endfor %}
          </div>
          <div class="mt-4">
            <a href="https://www.linkedin.com/in/samratk/" target="_blank" class="inline-flex items-center gap-2 px-4 py-2 bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg transition-all duration-300 text-white">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
              </svg>
              <span class="text-sm font-medium">Connect on LinkedIn</span>
            </a>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- About Section - Full Width -->
  <div class="px-8 pb-8">
    <div class="max-w-[80%] mx-auto">
      <div class="profile-card">
        <h2 class="section-title">About Me</h2>
        {% for paragraph in site.data.about.about.paragraphs %}
        <p class="text-gray-700 leading-relaxed mt-2 text-sm">{{ paragraph }}</p>
        {% endfor %}
      </div>
    </div>
  </div>

  <!-- Professional Experience with Right Panel -->
  <div class="px-8 pb-12">
    <div class="main-layout">
      
      <!-- Main Content Area -->
      <div class="main-content">
        
        <!-- Professional Experience Section -->
        <div class="mb-8">
          <div class="text-center mb-4">
            <h2 class="text-xl font-bold text-gray-900 mb-2">Professional Experience</h2>
            <div class="w-24 h-0.5 bg-gradient-to-r from-purple-500 to-violet-600 mx-auto rounded-full"></div>
          </div>

          <div class="space-y-4">
        
        {% for exp in site.data.about.experience %}
        <div class="experience-card">
          <div class="experience-header">
            <div class="flex-1">
              <h3 class="experience-title">{{ exp.title }}</h3>
              <p class="experience-company">{{ exp.company }}</p>
              {% if exp.department %}
              <p class="experience-department">{{ exp.department }}</p>
              {% endif %}
            </div>
            <div class="experience-duration">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              {{ exp.duration }}
            </div>
          </div>
          
          <div class="mb-4">
            <h4 class="font-bold text-gray-900 mb-3 text-base">Key Achievements & Responsibilities</h4>
            <ul class="achievement-list">
              {% for achievement in exp.achievements %}
              <li>{{ achievement }}</li>
              {% endfor %}
            </ul>
          </div>
        </div>
        {% endfor %}

        <!-- Honeywell Intelligrated -->
        <div class="experience-card">
          <div class="experience-header">
            <div class="flex-1">
              <h3 class="experience-title">Software Engineering Manager</h3>
              <p class="experience-company">Honeywell Intelligrated</p>
              <p class="experience-department">Warehouse Execution Systems - Multi-site Microservice Cloud Platform</p>
            </div>
            <div class="experience-duration">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              October 2021 - May 2023
            </div>
          </div>
          
          <div class="mb-4">
            <h4 class="font-bold text-gray-900 mb-3 text-lg">Key Achievements & Responsibilities</h4>
            <ul class="achievement-list">
              <li>Reorganized global software engineering teams across multiple sites, achieving <span class="highlight-metric">25% increase in development velocity</span> and <span class="highlight-metric">20% improvement in quality performance</span>, with consistently >85% stakeholder satisfaction</li>
              <li>Established Honeywell's <strong>offshore capability in Bangalore</strong>, successfully building and scaling a ~100-member strong engineering team by leveraging India's highly educated and digitally literate workforce</li>
              <li>Architected and mapped organization-wide <strong>Agile structure</strong> incorporating a sustained release train for scaled agile SDLC, driving <span class="highlight-metric">20% improvement in schedule compliance</span> across quarters and multiple customer releases</li>
              <li>Designed and deployed comprehensive <strong>end-to-end CI/CD automation framework</strong> running over <span class="highlight-metric">6,000+ regression tests</span> using Selenium/Cucumber/Java stack, resulting in <span class="highlight-metric">20% reduction in post-release defects</span> YoY and <span class="highlight-metric">50% performance improvement</span> at scale</li>
              <li>Led complete <strong>end-to-end design and test strategy</strong> for microservice-based applications, improving scoping, planning, and predictability of program execution. Reduced testing EAC by <span class="highlight-metric">15%</span> and uncovered major scalability risks upfront</li>
              <li>Incorporated <strong>design for resiliency</strong> patterns ensuring <span class="highlight-metric">99.99% availability</span> metrics for multi-site, multi-user web applications on private cloud infrastructure</li>
              <li>Optimized engineering practices by leading improvements in automation, infrastructure, processes, and team organization, reducing sprint-on-sprint volatility by <span class="highlight-metric">20%</span> and improving schedule and cost performance by <span class="highlight-metric">25%</span></li>
              <li>Built robust <strong>onshore customer-facing capability</strong> in Mason, Ohio, with competent team and automation framework for customer demos with software-hardware integration</li>
              <li>Supervised and built high-performing test team of <strong>30+ senior software engineers</strong>, delivering multi-user web applications for major customers including <strong>Target, The Home Depot, Big Lots, Hamilton, Nike</strong></li>
              <li>Invented and patented <strong>engine relight visualization methods and systems</strong> for aircraft avionics (Patent # US 11,385,072)</li>
            </ul>
          </div>
        </div>

        <!-- Honeywell - Senior Supervisor (Warehouse) -->
        <div class="experience-card">
          <div class="experience-header">
            <div class="flex-1">
              <h3 class="experience-title">Senior Software Engineering Supervisor</h3>
              <p class="experience-company">Honeywell Intelligrated</p>
              <p class="experience-department">Warehouse Process Automation</p>
            </div>
            <div class="experience-duration">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              October 2019 - September 2021
            </div>
          </div>
          
          <div class="mb-4">
            <h4 class="font-bold text-gray-900 mb-3 text-lg">Key Achievements & Responsibilities</h4>
            <ul class="achievement-list">
              <li>Delivered <strong>end-to-end warehouse process automation solution</strong> capable of fulfilling <span class="highlight-metric">5,000 orders per hour</span> with system reliability of <span class="highlight-metric">99.99%</span> for goods-to-person solution</li>
              <li>Led major infrastructure <strong>upgrade from manual Docker Compose deployments to Kubernetes orchestrated multi-node clusters</strong> with high availability configurations, enabling <span class="highlight-metric">40% improvement</span> in system availability</li>
              <li>Designed and deployed comprehensive <strong>VPT workflow simulation</strong> using JMeter stack to perform endurance testing for 2+ weeks continuously, establishing performance baselines for customer deployments with <span class="highlight-metric">50% improvement</span> in endurance metrics</li>
              <li>Modernized VPT simulation to <strong>scriptless test environment</strong> using Doppelio, reducing VPT readiness lead time by <span class="highlight-metric">70%</span></li>
              <li>Configured and tuned <strong>Kubernetes cluster topologies</strong>, component instances, and resource allocations to accommodate 5,000+ orders fulfillment per hour through reduced latency and increased scalability</li>
              <li>Implemented <strong>real-time monitoring</strong> of endurance test metrics using InfluxDB time-series database with Grafana dashboards for continuous performance insights</li>
              <li>Supervised globally distributed team of <strong>10+ senior software engineers</strong> developing microservice-based web applications</li>
            </ul>
          </div>
        </div>

        <!-- Honeywell Aerospace - Senior Supervisor (FMS) -->
        <div class="experience-card">
          <div class="experience-header">
            <div class="flex-1">
              <h3 class="experience-title">Senior Software Engineering Supervisor</h3>
              <p class="experience-company">Honeywell Aerospace</p>
              <p class="experience-department">Flight Management Systems - Avionics</p>
            </div>
            <div class="experience-duration">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              September 2014 - September 2019
            </div>
          </div>
          
          <div class="mb-4">
            <h4 class="font-bold text-gray-900 mb-3 text-lg">Key Achievements & Responsibilities</h4>
            <ul class="achievement-list">
              <li>Led major <strong>architectural transformation</strong> of fuel performance computing module from monolithic architecture to <strong>microservice REST API</strong> based on cloud, enabling brand new preflight planning solution for pilots</li>
              <li>Built industry's first Flight Management System for <strong>Boeing 747-8</strong> from scratch using object-oriented principles with MVC pattern, creating the industry's first FMS with SPL architecture and reducing development efforts by <span class="highlight-metric">40%</span></li>
              <li>Developed comprehensive <strong>end-to-end simulation system</strong> to run flight management on Windows laptop by building platform abstraction layer on Windows kernel, reducing testing efforts by <span class="highlight-metric">70%</span></li>
              <li>Successfully <strong>transitioned from Waterfall to Agile</strong> project management methodologies, reducing cycle time by <span class="highlight-metric">50%</span> for projects requiring 75+ days of development effort for <strong>Gulfstream, Airbus, Boeing, and Embraer</strong></li>
              <li>Delivered critical avionics features for major OEMs ensuring DO-178B compliance and certification standards</li>
            </ul>
          </div>
        </div>

        <!-- Technology Specialist -->
        <div class="experience-card">
          <div class="experience-header">
            <div class="flex-1">
              <h3 class="experience-title">Technology Specialist</h3>
              <p class="experience-company">Honeywell Aerospace</p>
              <p class="experience-department">Flight Management Systems - Core Platform</p>
            </div>
            <div class="experience-duration">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              September 2011 - September 2014
            </div>
          </div>
          
          <div class="mb-4">
            <h4 class="font-bold text-gray-900 mb-3 text-lg">Key Achievements & Responsibilities</h4>
            <ul class="achievement-list">
              <li>Designed and implemented <strong>common framework and platform layer</strong> for flight management systems, creating scalable and reusable architecture across multiple product lines built upon core assets</li>
              <li>Built <strong>rule-based state machine transition framework</strong> for flight phase transitions and Required Navigation Performance (RNP) transitions for FMS</li>
              <li>Developed <strong>native logging system</strong> for embedded software enabling comprehensive debugging and diagnostics capabilities</li>
              <li>Implemented <strong>Noise Abatement feature</strong> for Embraer aircraft on native flight management system, ensuring compliance with environmental regulations</li>
              <li>Developed <strong>RTA (Required Time of Arrival)</strong> feature for Boeing commercial jets, enhancing precision approach capabilities</li>
              <li>Implemented <strong>onboard-to-offboard application communication</strong> via onboard computer from cockpit to cloud, enabling real-time data exchange for microservice-based flight crew services</li>
              <li>Designed and implemented <strong>COM-based distributed simulation tool</strong> for monitoring messages in avionics bus, improving system integration testing</li>
            </ul>
          </div>
        </div>

        <!-- Early Career Summary -->
        <div class="experience-card">
          <div class="experience-header">
            <div class="flex-1">
              <h3 class="experience-title">Early Career Progression</h3>
              <p class="experience-company">Honeywell Aerospace</p>
              <p class="experience-department">Flight Management Systems & Avionics</p>
            </div>
            <div class="experience-duration">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              July 2003 - August 2011
            </div>
          </div>
          
          <div>
            <p class="text-gray-700 mb-3">Progressive career advancement through technical excellence and leadership:</p>
            <ul class="achievement-list">
              <li><strong>Tech Lead (January 2006 - May 2011):</strong> Led technical teams in developing critical FMS features and establishing software development best practices</li>
              <li><strong>Senior Software Engineer (July 2004 - December 2005):</strong> Contributed to major avionics system development with focus on performance optimization</li>
              <li><strong>Software Engineer (July 2003 - June 2004):</strong> Started career developing embedded avionics software with focus on DO-178B certified systems</li>
            </ul>
          </div>
        </div>

          </div>
        </div>
      
      </div>
      
      <!-- Right Panel -->
      <div class="right-panel">
        <div class="right-panel-sticky">
          
          <!-- Education -->
          <div class="profile-card">
            <h2 class="section-title">Education</h2>
            <div class="mt-3 space-y-3">
              {% for edu in site.data.about.education %}
              <div class="border-l-3 border-{{ edu.color }}-500 pl-3">
                <h3 class="font-bold text-gray-900 text-sm">{{ edu.degree }}</h3>
                <p class="text-gray-600 text-xs">{{ edu.institution }} • {{ edu.year }}</p>
              </div>
              {% endfor %}
            </div>
          </div>

          <!-- Publications -->
          <div class="profile-card">
            <h2 class="section-title">Publications</h2>
            {% for pub in site.data.about.publications %}
            <div class="mt-3">
              <h3 class="font-bold text-gray-900 text-sm">{{ pub.icon }} {{ pub.title }}</h3>
              {% if pub.date %}
              <p class="text-gray-600 text-xs">{{ pub.date }}</p>
              {% endif %}
              {% if pub.link %}
              <a href="{{ pub.link }}" target="_blank" class="text-purple-600 hover:text-purple-800 text-xs font-medium">
                {% if pub.type == 'book' %}View on Amazon →{% else %}View Details →{% endif %}
              </a>
              {% endif %}
              {% if pub.number %}
              <p class="text-xs text-gray-500">Patent # {{ pub.number }}</p>
              {% endif %}
            </div>
            {% endfor %}
          </div>

          <!-- Key Skills -->
          <div class="profile-card">
            <h2 class="section-title">Key Skills</h2>
            <div class="mt-3 space-y-2">
              {% for category in site.data.about.skills %}
              <div>
                <h3 class="font-semibold text-gray-900 text-xs mb-1">{{ category.category }}</h3>
                <div class="flex flex-wrap gap-1">
                  {% for skill in category.items %}
                  <span class="badge badge-pink text-xs px-2 py-1">{{ skill }}</span>
                  {% endfor %}
                </div>
              </div>
              {% endfor %}
            </div>
          </div>

          <!-- Certifications -->
          <div class="profile-card">
            <h2 class="section-title">Certifications</h2>
            <div class="mt-3 space-y-1 text-xs">
              {% for cert in site.data.about.certifications %}
              <div class="flex items-start gap-2">
                <span class="text-purple-600 text-sm">✓</span>
                <span class="text-gray-700">{{ cert }}</span>
              </div>
              {% endfor %}
            </div>
          </div>
          
        </div>
      </div>
      
    </div>
