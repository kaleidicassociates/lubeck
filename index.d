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
        $(TDNW $(MREF kaleidic, lubeck))
        $(TD First API that uses garbage collection for memory managment.)
    )
    $(TR
        $(TDNW $(MREF kaleidic, lubeck2))
        $(TD Second implementation that uses reference counted matrices and vectors.)
    )
)

$(BR)
$(BR)

$(MREF kaleidic, lubeck) and $(MREF kaleidic, lubeck2) doesn't provide the same functionality or API, although many things are similar.

Macros:
        TITLE=Lubeck
        WIKI=Lubeck
        DDOC_BLANKLINE=
        _=
