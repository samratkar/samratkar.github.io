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
  }
  
  .gradient-bg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }
  
  .profile-card {
    background: white;
    border-radius: 1rem;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
  }
  
  .profile-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
  }
  
  .section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #667eea;
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
    width: 200px;
    height: 200px;
    border-radius: 50%;
    border: 5px solid white;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    object-fit: cover;
  }
</style>

<div class="bg-gray-50 min-h-screen">
  <!-- Hero Section -->
  <div class="gradient-bg text-white py-16">
    <div class="max-w-7xl mx-auto px-6">
      <div class="flex flex-col md:flex-row items-center gap-8">
        <img src="/assets/img/my-photo-small.png" alt="Samrat Kar" class="profile-photo">
        <div class="text-center md:text-left">
          <h1 class="text-5xl font-bold mb-4">Samrat Kar</h1>
          <p class="text-xl text-blue-100 mb-4">Software Engineering Manager | AI/ML Tech Leader</p>
          <p class="text-lg opacity-90">Jeppesen Foreflight (Formerly Boeing) | Flight Efficiency Solutions</p>
          <div class="flex flex-wrap gap-3 mt-6 justify-center md:justify-start">
            <span class="badge badge-purple">23+ Years Experience</span>
            <span class="badge badge-purple">AI & Cloud Expert</span>
            <span class="badge badge-purple">Aviation Technology</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Stats Section -->
  <div class="max-w-7xl mx-auto px-6 -mt-8 mb-12">
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
      <div class="stat-card">
        <div class="stat-number">23+</div>
        <div class="stat-label">Years of Experience</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">3</div>
        <div class="stat-label">Major Organizations</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">100+</div>
        <div class="stat-label">Team Members Led</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">1</div>
        <div class="stat-label">Patent Holder</div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto px-6 pb-12">
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
      
      <!-- Left Column -->
      <div class="lg:col-span-2 space-y-8">
        
        <!-- About Section -->
        <div class="profile-card">
          <h2 class="section-title">About Me</h2>
          <p class="text-gray-700 leading-relaxed mt-4">
            Tech leader with 23 years of experience, specializing in developing cloud-native AI-based applications and on-board avionics systems for OEMs and airlines. Currently building agentic and multi-agent systems for AI-enabled services based on aviation data at Boeing/Jeppesen.
          </p>
          <p class="text-gray-700 leading-relaxed mt-3">
            I'm passionate about leveraging cutting-edge AI/ML technologies to solve complex aviation challenges, with a proven track record of delivering 1-2% fuel savings per flight through predictive deep learning models.
          </p>
        </div>

        <!-- Core Values -->
        <div class="profile-card">
          <h2 class="section-title">Core Values</h2>
          <div class="flex flex-wrap gap-3 mt-4">
            <span class="badge badge-purple text-lg">💪 Courage</span>
            <span class="badge badge-purple text-lg">🙏 Faith</span>
            <span class="badge badge-purple text-lg">❤️ Love</span>
          </div>
        </div>

        <!-- Work Experience -->
        <div class="profile-card">
          <h2 class="section-title">Work Experience</h2>
          <div class="mt-6 space-y-6">
            
            <div class="timeline-item">
              <div class="timeline-dot"></div>
              <h3 class="text-xl font-bold text-gray-900">Software Engineering Manager</h3>
              <p class="text-purple-600 font-semibold">Boeing (Jeppesen Foreflight) • May 2023 - Present</p>
              <p class="text-gray-600 text-sm mb-3">Flight Efficiency Solutions</p>
              <ul class="list-disc list-inside text-gray-700 space-y-2 text-sm">
                <li>Building agentic and multi-agent systems for AI-enabled aviation data services</li>
                <li>Built Flight Deck Advisor - Predictive DL model achieving 1-2% fuel savings per flight</li>
                <li>Led cloud migration of three flight operations products to Azure</li>
                <li>Implemented DataMesh integration on Databricks with automated ETL pipelines</li>
                <li>AS9100 / CMMI-5 expert</li>
              </ul>
            </div>

            <div class="timeline-item">
              <div class="timeline-dot"></div>
              <h3 class="text-xl font-bold text-gray-900">Software Engineering Manager</h3>
              <p class="text-purple-600 font-semibold">Honeywell Intelligrated • Oct 2021 - May 2023</p>
              <p class="text-gray-600 text-sm mb-3">Warehouse Execution Systems</p>
              <ul class="list-disc list-inside text-gray-700 space-y-2 text-sm">
                <li>Reorganized global teams, achieving 25% velocity increase and 20% quality improvement</li>
                <li>Built 100-member offshore engineering team in Bangalore</li>
                <li>Designed E2E CI/CD automation framework with 6000+ regression tests</li>
                <li>Achieved 99.99% availability for microservice-based multi-site applications</li>
              </ul>
            </div>

            <div class="timeline-item">
              <div class="timeline-dot"></div>
              <h3 class="text-xl font-bold text-gray-900">Senior Software Engineering Supervisor</h3>
              <p class="text-purple-600 font-semibold">Honeywell • Oct 2019 - Sep 2021</p>
              <ul class="list-disc list-inside text-gray-700 space-y-2 text-sm">
                <li>Delivered end-to-end warehouse automation solution handling 5000 orders/hour</li>
                <li>Upgraded from Docker Compose to Kubernetes, improving availability by 40%</li>
                <li>Improved endurance testing performance by 50% with Jmeter stack</li>
              </ul>
            </div>

            <div class="timeline-item">
              <div class="timeline-dot"></div>
              <h3 class="text-xl font-bold text-gray-900">Senior Software Engineering Supervisor</h3>
              <p class="text-purple-600 font-semibold">Honeywell Aerospace • Sep 2014 - Sep 2019</p>
              <p class="text-gray-600 text-sm mb-3">Flight Management Systems</p>
              <ul class="list-disc list-inside text-gray-700 space-y-2 text-sm">
                <li>Re-factored FMS from monolithic to microservice REST API architecture</li>
                <li>Built industry's first FMS with SPL architecture for B747-8, reducing dev efforts by 40%</li>
                <li>Transitioned from waterfall to agile, reducing cycle time by 50%</li>
                <li>Patent holder: Engine relight visualization for aircraft avionics (US 11,385,072)</li>
              </ul>
            </div>

            <div class="timeline-item">
              <h3 class="text-lg font-bold text-gray-900">Technology Specialist & Previous Roles</h3>
              <p class="text-purple-600 font-semibold">Honeywell Aerospace • Jul 2003 - Sep 2014</p>
              <p class="text-gray-700 text-sm mt-2">Progressed through roles: SW Engineer → Sr. SW Engineer → Tech Lead → Technology Specialist</p>
            </div>
          </div>
        </div>

        <!-- Education -->
        <div class="profile-card">
          <h2 class="section-title">Education</h2>
          <div class="mt-4 space-y-4">
            <div class="border-l-4 border-purple-500 pl-4">
              <h3 class="font-bold text-gray-900">PhD in Generative AI</h3>
              <p class="text-gray-600">Amrita University • In Progress</p>
            </div>
            <div class="border-l-4 border-blue-500 pl-4">
              <h3 class="font-bold text-gray-900">MTech in Cloud Computing</h3>
              <p class="text-gray-600">BITS Pilani • 2024</p>
            </div>
            <div class="border-l-4 border-green-500 pl-4">
              <h3 class="font-bold text-gray-900">BE in Computer Science</h3>
              <p class="text-gray-600">VTU • 2002</p>
            </div>
          </div>
        </div>

        <!-- Publications -->
        <div class="profile-card">
          <h2 class="section-title">Publications</h2>
          <div class="mt-4">
            <h3 class="font-bold text-gray-900">📚 Simoni's Story</h3>
            <p class="text-gray-600 text-sm">Published: December 2012</p>
            <a href="https://www.amazon.in/Simonis-Story-Samrat-Kar-ebook/dp/B00AK3G8AS/" target="_blank" class="text-purple-600 hover:text-purple-800 text-sm font-medium">View on Amazon →</a>
          </div>
          <div class="mt-4">
            <h3 class="font-bold text-gray-900">🔬 Patent</h3>
            <p class="text-gray-600 text-sm">Engine relight visualization methods and systems for aircraft avionics systems</p>
            <p class="text-sm text-gray-500">Patent # US 11,385,072</p>
          </div>
        </div>
      </div>

      <!-- Right Column -->
      <div class="space-y-8">
        
        <!-- Contact -->
        <div class="profile-card">
          <h2 class="section-title">Contact</h2>
          <div class="mt-4 space-y-3">
            <div class="flex items-start gap-3">
              <svg class="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path>
              </svg>
              <span class="text-gray-700">Bangalore, India</span>
            </div>
            <div class="flex items-start gap-3">
              <svg class="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"></path>
              </svg>
              <span class="text-gray-700">+91-9342554680</span>
            </div>
            <div class="flex items-start gap-3">
              <svg class="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
              </svg>
              <span class="text-gray-700">samratk@gmail.com</span>
            </div>
          </div>
        </div>

        <!-- Social Links -->
        <div class="profile-card">
          <h2 class="section-title">Connect</h2>
          <div class="mt-4 space-y-3">
            <a href="https://www.linkedin.com/in/samratk/" target="_blank" class="social-link block text-center">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
              </svg>
              LinkedIn
            </a>
            <a href="https://x.com/samrat_kar" target="_blank" class="social-link block text-center">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
              </svg>
              Twitter / X
            </a>
            <a href="https://karconversations.wordpress.com/" target="_blank" class="social-link block text-center">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm0 18.5c-4.687 0-8.5-3.813-8.5-8.5S7.313 3.5 12 3.5s8.5 3.813 8.5 8.5-3.813 8.5-8.5 8.5zm-1.354-5.646l-2.646-2.647c-.195-.195-.195-.512 0-.707s.512-.195.707 0l2.292 2.293 5.293-5.293c.195-.195.512-.195.707 0s.195.512 0 .707l-5.646 5.647c-.196.195-.512.195-.707 0z"/>
              </svg>
              Blog
            </a>
          </div>
        </div>

        <!-- Passions -->
        <div class="profile-card">
          <h2 class="section-title">Passions</h2>
          <div class="mt-4 space-y-2">
            <div class="badge badge-blue w-full text-left">🧠 Positive Psychology</div>
            <div class="badge badge-blue w-full text-left">👨‍💼 Leadership Development</div>
            <div class="badge badge-blue w-full text-left">🤖 AI and Machine Learning</div>
            <div class="badge badge-blue w-full text-left">📖 Scriptures & Philosophy</div>
            <div class="badge badge-blue w-full text-left">🧘 Hatha Yoga & Meditation</div>
            <div class="badge badge-blue w-full text-left">🤝 Reaching out to people</div>
          </div>
        </div>

        <!-- Hobbies -->
        <div class="profile-card">
          <h2 class="section-title">Hobbies</h2>
          <div class="mt-4 space-y-2">
            <div class="badge badge-green w-full text-left">🔬 STEM Education for underprivileged</div>
            <div class="badge badge-green w-full text-left">📚 Scripture reading for children</div>
            <div class="badge badge-green w-full text-left">🎵 Music</div>
          </div>
        </div>

        <!-- Key Skills -->
        <div class="profile-card">
          <h2 class="section-title">Key Skills</h2>
          <div class="mt-4 space-y-3">
            <div>
              <h3 class="font-semibold text-gray-900 text-sm mb-2">AI & ML</h3>
              <div class="flex flex-wrap gap-2">
                <span class="badge badge-pink text-xs">Agentic RAG</span>
                <span class="badge badge-pink text-xs">Multi-Agents</span>
                <span class="badge badge-pink text-xs">Azure OpenAI</span>
                <span class="badge badge-pink text-xs">Deep Learning</span>
              </div>
            </div>
            <div>
              <h3 class="font-semibold text-gray-900 text-sm mb-2">Programming</h3>
              <div class="flex flex-wrap gap-2">
                <span class="badge badge-pink text-xs">Python</span>
                <span class="badge badge-pink text-xs">Java</span>
                <span class="badge badge-pink text-xs">C/C++</span>
                <span class="badge badge-pink text-xs">JavaScript</span>
              </div>
            </div>
            <div>
              <h3 class="font-semibold text-gray-900 text-sm mb-2">DevOps & Cloud</h3>
              <div class="flex flex-wrap gap-2">
                <span class="badge badge-pink text-xs">Kubernetes</span>
                <span class="badge badge-pink text-xs">Docker</span>
                <span class="badge badge-pink text-xs">Azure</span>
                <span class="badge badge-pink text-xs">Databricks</span>
              </div>
            </div>
            <div>
              <h3 class="font-semibold text-gray-900 text-sm mb-2">Architecture</h3>
              <div class="flex flex-wrap gap-2">
                <span class="badge badge-pink text-xs">Microservices</span>
                <span class="badge badge-pink text-xs">Cloud Native</span>
                <span class="badge badge-pink text-xs">Big Data</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Certifications -->
        <div class="profile-card">
          <h2 class="section-title">Certifications</h2>
          <div class="mt-4 space-y-2 text-sm">
            <div class="flex items-start gap-2">
              <span class="text-purple-600">✓</span>
              <span class="text-gray-700">Scrum & SAFe Certified</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-purple-600">✓</span>
              <span class="text-gray-700">Google Certified Project Management</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-purple-600">✓</span>
              <span class="text-gray-700">AI-Machine Learning (IIIT Bangalore)</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-purple-600">✓</span>
              <span class="text-gray-700">Executive MBA Certificate (IIM Bangalore)</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-purple-600">✓</span>
              <span class="text-gray-700">Positive Psychology</span>
            </div>
          </div>
        </div>

        <!-- Inspirations -->
        <div class="profile-card">
          <h2 class="section-title">Inspirations</h2>
          <p class="text-gray-600 text-sm mt-4">Learning from thought leaders in psychology, leadership, and philosophy including Brene Brown, Viktor Frankl, Simon Sinek, Stephen Covey, Rabindranath Tagore, Swami Vivekananda, and many others.</p>
        </div>
      </div>
    </div>
  </div>
</div>

## Passions

1. Positive Psychology
2. Leadership Development
3. AI and Machine Learning
4. Scriptures and Philosophy
5. Hatha Yoga
6. Meditation
7. Reaching out to people

## 3 Core Values

1. Courage
2. Faith
3. Love

## Masters

1. Brene Brown
2. Alfred Adler
3. Viktor Frankl
4. Erich Fromm
5. Czsikszentmihalyi Mihaly
6. Martin Seligman
7. Daniel Goleman
8. John Maxwell
9. Simon Sinek
10. Stephen Covey
11. Jim Collins
12. Peter Drucker
13. Daniel Kahneman
14. Daniel Pink
15. Malcolm Gladwell
16. Rabindranath Tagore
17. Leo Tolstoy
18. Swami Vivekananda
19. Sir John Woodroffe

## Education

1. BE Computer Science - VTU - 2002
2. MTech Cloud Computing - BITS Pilani - 2024
3. PhD Generative AI - Amrita University - In Progress

## Businesses worked for

1. Honeywell Aerospace - 2003 - 2021
2. Honeywell Intelligrated - 2021 - 2023
3. Boeing Digital Aviation Services (Jeppesen) - 2023 - present

## Hobbies

1. STEM Education for underprivileged children
2. Reading of scriptures for school children
3. Music

## Reach out to me at :

1. [https://www.linkedin.com/in/samratk/](https://www.linkedin.com/in/samratk/)
2. [https://samratkar.github.io/](https://samratkar.github.io/)
3. [https://x.com/samrat_kar](https://x.com/samrat_kar)
4. [https://karconversations.wordpress.com/](https://karconversations.wordpress.com/)
5. [https://criativ-mind.blogspot.com/](https://criativ-mind.blogspot.com/)

## My Books published

1. [Simoni's Story](https://www.amazon.in/Simonis-Story-Samrat-Kar-ebook/dp/B00AK3G8AS/) - Dec 2012


Samrat Kar
software engineering manager | Jeppesen Foreflight (Formerly Boeing)| flight efficiency solutions
tech leader with 23 years of experience, with a track record of developing cloud-native AI-based applications and on-board avionics systems for OEMs and airlines.
WORK EXPERIENCE
______________________________________________________________________

Flight efficiency solutions. Boeing.                                      
i. software engineering manager	5/22/2023 - Present
building agentic and multi-agent systems for AI-enabled services based on aviation data.  
building and leading prescriptive Deep learning model for fuel efficiency for commercial and military aircrafts - Built a product known as Flight Deck Advisor, which predicts the most optimum cost index to enable the most efficient fuel flow for any aircraft. This solution is fleet agnostic and results in 1-2% fuel savings per flight.
built and led teams for real-time wind and route uplink to Flight Deck for both commercial and business jets. 
cloud migration—migration of three flight operations products—wind updates, route sync, and flight deck advisor from on-prem to Azure. 
datamesh integration—porting the ETL pipeline into datamesh implemented on databricks. Have set up an automated ETL pipeline to ingest, process, and visualise aircraft Quick Access Recorder (QAR) flight data. 
AS9100 / CMMI-5 expert

Warehouse execution system. Honeywell.
multi-site, microservice, private cloud-based software + hardware system. 

i. Software engineering manager	                                        10/2021–5/12/2023
reorganised global software engineering teams to grow development velocity by 25% & improved quality performance by 20%, and consistently  >85% stakeholder satisfaction. 
created the firm’s offshore capability in Bangalore to build a ~100-member strong engineering team. Took advantage of low marginal employment rates, a well-educated & highly digitally literate workforce.
mapped org-wide agile structure for incorporating a sustained release train for a scaled agile SDLC, across quarters and releases to multiple customers, driving improvement in schedule compliance by 20%.
designed and deployed e2e CICD automation fwk running over 6000 regression tests by constructing a selenium/cucumber/java framework resulting in 20% reduced post-release defects year on year and 50% performance improvement at scale.
designed and deployed the entire end-to-end design and test strategy for microservice-based applications, leading to improved scoping, planning, and predictability of program execution. This helped reduce the EAC of testing by 15% and helped uncover some major risks towards scalability and volume performance of the design upfront. 
incorporated  design for resiliency to ensure 99.99% availability metrics of a micro service based multi-site, multi-user web application on private cloud.
optimized engineering practices by leading improvements in automation, infrastructure, processes, team practices and team  organization. This reduced the sprint on sprint volatility by 20% leading to improvement in schedule and cost performance by 25% across releases.
created the firm's onshore capability by building up a competent onshore customer facing team and a robust process and automation to drive customer demos with sw-hw integration, in Mason, Ohio, US. 
built a robust competency improvement framework across design and test team to build a sustainable pipeline of talent acquisition, upskilling, succession planning, and long term career growth, across global sites.
supervised and built a high performing test team of 30+ senior software engineers, to test and deliver a multi user web application running on a private cloud, for multiple customers including Target, THD, Biglots, Hamilton, Nike, for multiple releases.
have a patent on engine relight visualization methods and systems for aircraft avionics systems. patent # US 11,385,072


ii. senior software engineering supervisor	                         10/2019 – 09/2021

delivered end to end warehouse process automation solution to fulfill 5000 orders per hour, with system reliability of 99.99% for goods to person solution of micro service based web app running on private cloud, orchestrated on kubernetes with high availability configuration. 
upgraded from manual docker compose based deployments to kubernetes orchestrated multi-node clusters, with high availability configurations, enabling improvement of the system availability by 40%. 
designed and deployed an e2e warehouse process orchestration tool for VPT workflow simulation using jmeter stack to perform endurance testing for a time period of two weeks and beyond. This enabled baselining of volume and performance metrics for customer site deployments. This resulted in 50% improvement on endurance metrics performance. 
improvised the VPT simulation stack to a more scriptless test environment using a third party tool - doppelio. This helped reducing VPT readiness lead time by 70%
configured and tuned k8 cluster topologies, component instances, to accommodate the demands of 5000+ orders fulfillment per hour, by reduced latency, increasing scalability and availability. 
designed and deployed monitoring of real time endurance test metrics on a daily basis using time series influx database with grafana.
supervised a team of 10+ senior software engineers globally for micro service web app.

Flight management systems. Honeywell.
avionics system for flight planning, navigation, perf computations.
iii. senior software engineering supervisor                              09/2014 – 09/2019
re-factored fuel performance computing module of onboard avionics system - flight management system (fms) from monolithic to micro service REST API based on cloud, enabling a brand new solution for pilot preflight planning.
built an e2e simulation system to run flight management on windows laptop, by building a platform abstraction layer on windows kernel. This reduced testing efforts by 70%
built a flight management system from scratch for B747-8 using object oriented principles with MVC pattern. This led to the industry's first FMS created with SPL architecture. Helped reduce dev efforts by 40%
Transitioned from waterfall to agile project management methods, reducing 50% cycle time for projects requiring 75+ days of development effort for Glufstream, Airbus, Boeing and Embraer. 
iv. technology specialist                                                              09/2011–09/2014
designed and implemented a common framework and platform layer for flight management systems. This was a scalable and reusable architecture across multiple product lines, building upon core assets. 
Built a rule-based state machine transition framework for flight phase transitions and required navigation performance transitions for FMS. 
developed a native logging system for embedded software
Implemented noise abatement feature for Embraer on the native flight mgmt system app in the onboard avionics. 
implemented RTA (return to arrival feature) for boeing commercial jets. 
Implemented onboard to off-board application communication via onboard computer from cockpit to the cloud to support micro service based services for flight crew.
designed and implemented a COM-based distributed simulation tool for monitoring messages in the avionics bus. 
PREVIOUS EXPERIENCE
______________________________________________________________________

v. tech Lead - Honeywell, 01/2006–05/2011
vi. sr. sw engineer - Honeywell, 07/2004–12/2005
vii. SW Engineer - Honeywell, 07/2003–06/2004
CONTACT
__________________________

•  Bangalore, India
•  +91-9342554680
•  samratk@gmail.com
SKILLS
__________________________

Hard Skills: 
•   AI -
 Agentic RAG, multi-agents, Azure Open ai, open, building end-to-end chatbots, deep learning ensemble models for predictive analytics. 
•   programming - 
    c, c++, java,  python,
   javascript, shell script, 
   linux system prog, 
   mpi, openmp, pthreads, 
   map reduce in python.
•   testing - 
     cucumber, jmeter,
    selenium, junit, mockito, 
    postman
•   devops - 
    dockers, kubernetes,
    jenkins, git, bitbucket,
    bamboo, azure, databricks
• build tools - 
      gradle, maven
• database, cache & queues - 
      postgres, mongo db, 
     rabbitmq, hazelcast,
     hadoop hdfs, map reduce
• telemetry -
     grafana, appdynamics,
      databricks
•   project mgmt - 
     jira portfolio, mpp, gantt
•   agile methodologies 
•   cloud computing - azure
•   microservice architecture
•   big data systems
•   parallel & distributed compute
•   computer networks
•   neural net & deep learning

Techniques: 
•   software development
•   product management
•   requirements analysis
•   system design
•   six sigma, cmmi-5, pcmmi-5 
•   value stream mapping
•   sparks, luma
•   design thinking
•   project mgmt
•   do178-b








EDUCATION
__________________________

•   PhD Generative Ai (in prog) 
•   MTech cloud computing 
    BITS Pilani
•   AI-machine learning - IIIT B
•   Executive MBA Cert - IIM B
•   BE computer science - VTU

OTHER CERTS
___________________________

•   scrum & SAFe certified
•   google certified project mgmt
•   positive psychology
•   estimation & planning
•   people management &
    leadership
•   conflict resolution




