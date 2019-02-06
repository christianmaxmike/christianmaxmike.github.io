---
layout: archive
title: "MindNotes"
permalink: /mindnotes/
author_profile: true
---

{% include base_path %}

The intention behind the series <i>MindNotes</i> is to give students a deep insight in not only the theory i present in my tutorials, but also to give a practical approach to it. Therefore, i try to give an easy access to the topics of diverse lectures for which i assist as a research assistant. The lectures cover topics on 'Big Data Management and Analytics', 'Knowledge Discovery and Data Mining' and 'Machine Learning'. 
This series will grow with time and should not be considered as being in a final state. I try to keep the topics as clean as possible and MathJax (javascript should be activated in your browser) is used to render the formulas. Sometimes there are some render issues for which i'm highly sorry (and for potential typos). Nonetheless, i hope you can profit from this series. So, let's get started!  

<hr>
<h2>Machine Learning - Unsupervised Learning - Clustering</h2>
<hr>
{% for post in site.mindnotes %}
    {% if post.topic == "ml-ul-clustering" %}
      {% include archive-single.html %}
    {% endif %}
{% endfor %}

<hr>
<h2>Deep Learning - Basics</h2>
<hr>
{% for post in site.mindnotes %}
    {% if post.topic == "dl-basics" %}
      {% include archive-single.html %}
    {% endif %}
{% endfor %}

<hr>
<h2>Data Mining - Sequence Mining</h2>
<hr>
{% for post in site.mindnotes %}
    {% if post.topic == "dm-sequenceMining" %}
      {% include archive-single.html %}
    {% endif %}
{% endfor %}

