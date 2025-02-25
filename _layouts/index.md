---
layout: default
title: Home
---

<div class="home">
  
  <ul class="post-list">
    {%- for post in site.posts -%}
      <li>
        <h2>
          <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
        </h2>
        <p style="text-align: justify">
          {{ post.description }}
        </p>
        <p>Debug Info: Title = {{ post.title }}, Description = {{ post.description }}</p>
      </li>
    {%- endfor -%}
  </ul>

</div>
