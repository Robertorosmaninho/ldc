
/* Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * All Rights Reserved, written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/dmd/root/rmem.h
 */

#pragma once

#include "dcompat.h"    // for d_size_t

struct Mem
{
    Mem() { }

    static char *xstrdup(const char *s);
    static void xfree(void *p);
    static void *xmalloc(d_size_t size);
    static void *xcalloc(d_size_t size, d_size_t n);
    static void *xrealloc(void *p, d_size_t size);
    static void error();

#if 1 // version (GC)
    static bool _isGCEnabled;

    static bool isGCEnabled();
    static void disableGC();
    static void addRange(const void *p, d_size_t size);
    static void removeRange(const void *p);
#endif
};

extern Mem mem;
