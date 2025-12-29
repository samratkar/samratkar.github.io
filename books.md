---
layout: default
title: Books
permalink: /books/
---

<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
  body {
    font-family: 'Inter', sans-serif;
    max-width: 100% !important;
    margin: 0 !important;
  }
  
  .gradient-bg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }
  
  .book-card {
    background: white;
    border-radius: 0.75rem;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
  }
  
  .book-card:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
  }
  
  .books-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1.5rem;
    margin-bottom: 3rem;
  }
  
  .published-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 1.5rem;
  }
  
  .header-image {
    width: 100%;
    max-width: 800px;
    margin: 2rem auto;
    display: block;
    border-radius: 0.75rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
  
  .section-title {
    color: #667eea;
    font-weight: 700;
    font-size: 1.875rem;
    margin-top: 3rem;
    margin-bottom: 1.5rem;
  }
  
  .book-title {
    color: #764ba2;
    font-weight: 600;
    font-size: 1.5rem;
    margin-bottom: 1rem;
  }
  
  .book-info {
    color: #4a5568;
    line-height: 1.8;
  }
  
  .book-info strong {
    color: #2d3748;
  }
  
  .book-link {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s ease;
  }
  
  .book-link:hover {
    color: #764ba2;
    text-decoration: underline;
  }
  
  .published-book {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
  }
  
  .published-book a {
    color: white;
    text-decoration: underline;
    font-weight: 500;
  }
  
  .published-book a:hover {
    opacity: 0.9;
  }
  
  .hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
  
  .hero-banner h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }
  
  .hero-banner p {
    font-size: 1.125rem;
    opacity: 0.95;
  }
  
  .back-home-link {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    color: #4a5568;
    text-decoration: none;
    font-weight: 500;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    transition: all 0.2s ease;
    margin: 1.5rem 0;
  }
  
  .back-home-link:hover {
    color: #667eea;
    background-color: #f7fafc;
  }
</style>

<div class="bg-gray-50 min-h-screen">
  
  <!-- Back to Home Link -->
  <div class="px-8 pt-6">
    <a href="/" class="back-home-link">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
      </svg>
      <span>Back to Home</span>
    </a>
  </div>
  
  <!-- Hero Banner -->
  <div class="hero-banner">
    <h1>{{ site.data.books.header.title }}</h1>
    <p>{{ site.data.books.header.subtitle }}</p>
  </div>
  
  <div class="max-w-6xl mx-auto px-8 py-8">
    <img src="{{ site.data.books.header.image }}" alt="Books Header" class="header-image">
  
  <h2 class="section-title">📚 Books I'm Reading</h2>
  
  <div class="books-grid">
    {% for book in site.data.books.reading_list %}
    <div class="book-card">
      <h3 class="book-title">{{ forloop.index }}. {{ book.title }}</h3>
      <div class="book-info" style="flex-grow: 1;">
        <p style="margin-bottom: 0.5rem;"><strong>Author:</strong> {{ book.author }}</p>
        <p style="margin-bottom: 0.5rem;"><strong>Started:</strong> {{ book.date_started }}</p>
        <p style="margin-bottom: 0.5rem;"><strong>Status:</strong> {{ book.date_completed }}</p>
        {% if book.review_url != 'TBD' %}
        <p style="margin-top: 1rem;"><a href="{{ book.review_url }}" class="book-link">📖 Read Review</a></p>
        {% endif %}
      </div>
    </div>
    {% endfor %}
  </div>
  
  <h2 class="section-title">✍️ My Published Books</h2>
  
  <div class="published-grid">
    {% for book in site.data.books.published_books %}
    <div class="published-book">
      <h3 style="margin-bottom: 0.5rem; font-size: 1.25rem;">{{ book.title }}</h3>
      <p style="margin: 0;">Published: {{ book.publication_date }} | <a href="{{ book.url }}" target="_blank">View on Amazon</a></p>
      {% if book.description %}
      <p style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.95;">{{ book.description }}</p>
      {% endif %}
    </div>
    {% endfor %}
  </div>
  
</div>
</div>