---
layout: archive
title: "MindNotes"
permalink: /mindnotes/
author_profile: true
---

{% include base_path %}

The intention behind the series <i>MindNotes</i> is to give my students a deeper insight in not only the theory i present in my courses, but also to give a practical approach to it. Therefore, i try to give an easy access to the topics of diverse lectures for which i assist as a research assistant. The lectures cover topics on 'Big Data Management and Analytics', 'Knowledge Discovery and Data Mining' and 'Machine Learning'. 
This series will grow with time and should not be considered as being in a final state. I try to keep the topics as clean as possible and MathJax (javascript should be activated in your browser) is used to render the formulas. Sometimes there are some render issues for which i'm highly sorry (and for potential typos). Nonetheless, i hope you can profit from this series. So, let's get started!  

<h1>Practical Training</h1>
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

<hr>
<h1>Theoretical Training</h1>
<h2>AI Essentials</h2>
<h4>Math Basics</h4>
<ul>
<li><a href="https://christianmaxmike.github.io/mindnotes/ai_math_derivatives.pdf">Derivatives</a></li>
</ul>
<h4>Deep Learning Basics</h4>
<ul>
<li><a href="https://christianmaxmike.github.io/mindnotes/ai_dl_activationFunctions.pdf">Activation Functions</a></li>
<li><a href="https://christianmaxmike.github.io/mindnotes/ai_dl_backpropagation.pdf">Backpropagation</a></li>
</ul>
<h4>Machine Learning Basics</h4>
--- coming soon ---

<!--<object data="{{ post.file_document_path }}" width="1000" height="1000" type='application/pdf'/>-->

