{% extends 'markdown/index.md.j2' %}

{% block header %}
---
layout: mermaid
title: "{{ resources.metadata.name | replace('_', ' ') }}"
date: {{ resources.metadata.modified_date if resources.metadata.modified_date else '2025-05-18' }}
---
{% endblock header %}

{% block body %}
{{ super() }}
{% endblock body %}
