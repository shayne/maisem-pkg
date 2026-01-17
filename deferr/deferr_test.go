// Copyright (c) 2025 AUTHORS All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deferr

import (
	"errors"
	"slices"
	"testing"
)

func TestCleanup(t *testing.T) {
	t.Run("no error", func(t *testing.T) {
		var called bool
		_ = func() (err error) {
			defer Cleanup(&err, func() error {
				called = true
				return nil
			})
			return nil
		}()
		if called {
			t.Error("cleanup should not be called when no error")
		}
	})

	t.Run("with error", func(t *testing.T) {
		var called bool
		cleanupErr := errors.New("cleanup error")
		err := func() (err error) {
			defer Cleanup(&err, func() error {
				called = true
				return cleanupErr
			})
			return errors.New("original error")
		}()
		if !called {
			t.Error("cleanup should be called when error")
		}
		// The original error is preserved, cleanup error is returned by Cleanup
		// but not captured since defer doesn't capture return values
		if err == nil || err.Error() != "original error" {
			t.Errorf("expected original error, got %v", err)
		}
	})
}

func TestClosers(t *testing.T) {
	t.Run("CloseOnErr with nil error", func(t *testing.T) {
		var called bool
		err := func() (err error) {
			var c Closers
			defer c.CloseOnErr(&err)
			c.Add(func() error {
				called = true
				return nil
			})
			return nil
		}()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if called {
			t.Error("closers should not be called when no error")
		}
	})

	t.Run("CloseOnErr with error", func(t *testing.T) {
		var called bool
		err := func() (err error) {
			var c Closers
			defer c.CloseOnErr(&err)
			c.Add(func() error {
				called = true
				return nil
			})
			return errors.New("test error")
		}()
		if err == nil {
			t.Error("expected error")
		}
		if !called {
			t.Error("closers should be called when error")
		}
	})

	t.Run("Close order is reversed", func(t *testing.T) {
		var order []string
		var c Closers
		c.Add(func() error { order = append(order, "a"); return nil })
		c.Add(func() error { order = append(order, "b"); return nil })
		c.Add(func() error { order = append(order, "c"); return nil })
		c.Close()
		if !slices.Equal(order, []string{"c", "b", "a"}) {
			t.Errorf("expected [c, b, a], got %v", order)
		}
	})

	t.Run("Close joins errors", func(t *testing.T) {
		var c Closers
		err1 := errors.New("error 1")
		err2 := errors.New("error 2")
		c.Add(func() error { return err1 })
		c.Add(func() error { return nil })
		c.Add(func() error { return err2 })
		err := c.Close()
		if err == nil {
			t.Fatal("expected error")
		}
		if !errors.Is(err, err1) {
			t.Errorf("expected error to contain err1")
		}
		if !errors.Is(err, err2) {
			t.Errorf("expected error to contain err2")
		}
	})
}
