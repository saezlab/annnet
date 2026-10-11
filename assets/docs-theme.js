/*
 * Shared behaviour of the saezlab MkDocs Material sites. Do not edit
 * the copy in a package repository: change the original in saezverse
 * (guidelines/docs-theme/) and copy it again.
 *
 * A click on a plot in a notebook shows it at full size over the page.
 * A click or the Escape key closes it.
 *
 * The navigation tabs in the header are not part of what Material
 * replaces on instant navigation (navigation.instant). After each page
 * load, the active tab is set from the active top-level item of the
 * side navigation: both list the same items in the same order.
 */
(function () {
  function syncTabs() {
    var items = document.querySelectorAll(
      '.md-nav--primary > .md-nav__list > .md-nav__item'
    );
    var tabs = document.querySelectorAll('.sv-header-tabs .md-tabs__item');

    if (items.length !== tabs.length) {
      return;
    }

    items.forEach(function (item, i) {
      tabs[i].classList.toggle(
        'md-tabs__item--active',
        item.classList.contains('md-nav__item--active')
      );
    });
  }

  if (window.document$) {
    window.document$.subscribe(syncTabs);
  }

  function close() {
    var zoom = document.querySelector('.sv-zoom');

    if (zoom) {
      zoom.remove();
    }
  }

  document.addEventListener('click', function (event) {
    var img = event.target.closest('.jupyter-wrapper .jp-RenderedImage img');

    if (event.target.closest('.sv-zoom')) {
      close();
      return;
    }

    if (!img) {
      return;
    }

    var zoom = document.createElement('div');
    var large = document.createElement('img');

    zoom.className = 'sv-zoom';
    large.src = img.src;
    large.alt = img.alt;
    zoom.appendChild(large);
    document.body.appendChild(zoom);
  });

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
      close();
    }
  });
})();
