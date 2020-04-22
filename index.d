Ddoc

$(H1 High level linear algebra library for Dlang)

$(P The following table is a quick reference guide for which Lubeck modules to
use for a given category of functionality.)

$(BOOKTABLE ,
    $(TR
        $(TH Modules)
        $(TH Description)
    )
    $(TR
        $(TDNW $(LINK2 lubeck.html, lubeck))
        $(TD First API that uses garbage collection for memory managment.)
    )
    $(TR
        $(TDNW $(LINK2 lubeck2.html, lubeck2))
        $(TD Second implementation that uses reference counted ndslices for memory managment.)
    )
)

$(LINK2 lubeck.html, lubeck) and $(LINK2 lubeck2.html, lubeck2) doesn't provide the same functionality or API, although many things are similar.

Macros:
        TITLE=Mir Optim
        WIKI=Mir Optim
        DDOC_BLANKLINE=
        _=
