---
layout: archive
title: "MindNotes"
permalink: /mindnotes/
author_profile: true
---

{% include base_path %}

The intention behind the series <i>MindNotes</i> is to give students a deep insight in not only the theory i present in my tutorials, but also a practical approach to it. Therefore, i try to give an easy access to the topics of diverse lectures for which i assist. The lectures cover topics on 'Big data management and Analytics', 'Knowledge Discovery and Data Mining' and 'Machine Learning'. 
This series will grow with time and should not be considered as final. I try to keep the topics as clean as possible and MathJax is used to render the formulas. Sometimes there are some render issues for which i'm highly sorry (and for potential typos). Nonetheless, i hope you can profit from this series. So, let's get started! 

<h3>Machine Learning - Unsupervised Learning - Clustering</h3>
<hr>
{% for post in site.mindnotes reversed %}
    {% if post.topic == "ml-ul-clustering" %}
      {% include archive-single.html %}
    {% endif %}
{% endfor %}
