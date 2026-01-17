// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package deferr provides helpers for cleaning up resources on error.
package deferr

import "errors"

// Cleanup calls f if *err is not nil and returns f's error.
// Use this for simple one-off cleanup in defer statements:
//
//	func NewComponent() (c *Component, err error) {
//	    r1, err := initResource1()
//	    if err != nil {
//	        return nil, err
//	    }
//	    defer Cleanup(&err, r1.Close)
//
//	    r2, err := initResource2()
//	    if err != nil {
//	        return nil, err
//	    }
//	    defer Cleanup(&err, r2.Close)
//	    ...
//	    return &Component{r1, r2}, nil
//	}
func Cleanup(err *error, f func() error) error {
	if *err == nil {
		return nil
	}
	return f()
}

// Closers collects cleanup functions to be called later.
// Use this when you need to reuse the closers (e.g., for a component's Close method):
//
//	func NewComponent() (c *Component, err error) {
//	    var closers Closers
//	    defer closers.CloseOnErr(&err)
//
//	    r1, err := initResource1()
//	    if err != nil {
//	        return nil, err
//	    }
//	    closers.Add(r1.Close)
//
//	    r2, err := initResource2()
//	    if err != nil {
//	        return nil, err
//	    }
//	    closers.Add(r2.Close)
//	    ...
//	    return &Component{closers: closers}, nil
//	}
//
//	func (c *Component) Close() error {
//	    return c.closers.Close()
//	}
//
// Closers is not safe for concurrent use.
type Closers struct {
	fs []func() error
}

// Add adds a cleanup function to be called later.
func (c *Closers) Add(f func() error) {
	c.fs = append(c.fs, f)
}

// Close calls all functions in reverse order and returns any errors joined.
func (c *Closers) Close() error {
	var errs []error
	for i := len(c.fs) - 1; i >= 0; i-- {
		if err := c.fs[i](); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// CloseOnErr calls Close if *err is not nil.
func (c *Closers) CloseOnErr(err *error) {
	if *err == nil {
		return
	}
	c.Close()
}
