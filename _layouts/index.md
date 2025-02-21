```markdown
---
layout: default
title: Home
---

# Welcome to My Portfolio

Below are some of my latest projects:

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> - {{ post.date | date: "%B %d, %Y" }}
    </li>
  {% endfor %}
</ul>
```
