(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("024ce446-433a-4b07-bad2-aaae5d675d6e");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '024ce446-433a-4b07-bad2-aaae5d675d6e' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"e8d63e5e-05a4-441a-841c-9be9b2db2e73":{"roots":{"references":[{"attributes":{"data":{"x":{"__ndarray__":"uFFFN8IjBcD7fUoHbAEFwGBsiwkJrAPA3Q9vBdmUAsDXgF7dx4T/v6ZLzeCBU/q/Pf78M1RT+b9yMObQnBj5v35CPxClFPa/vqAYm6UJ9r998ifgGcX1v6xlKX+a9fS/rabxjOm89L/wyz9pRw7xv4P6lzM2zfC/h/t5g6eJ8L9FYNv/yjnvv+Pr9mhWE+y/vu+DRgVc6r/Y8rTJ8fXpv85nmA+WkOm/n5yUVNjb579NL68F82DnvwzPbi9URue/lX/qJxI55797dgsXRQjnv1m6+tATs+a//PpL5YLw5L/0WSjbwKvjv37xmuJGq+O/0CLmVBd347/lOH+psDLjvxJLU4OKKeK/49OwzY7s4b8PRhBDcF7hv/U1OQkcqd6/t1Px7IH+278STgrpIBvYv7KfkBZv6de/GLubC5xx1r8GHB04gBXVv39+Hf//rdO/+gUOjDaH0r/BYcWZwNfMv07XuJew1Mq/KOEdvJeSyr+qaegZ/AXDvzJxKaIWjra/z18RR5dus79UjlSdKsarv4Sj8pXv2ai/PwNvyEIAqD8zevnsorSrP06XRHRajcA/+GaZQKCyxT/1rUFAj9zGP7TMAJbVDMc/LQ05Asomyz+fK3nBZtHPP/RzSarUsNA/S5UhCQNl0T93xAYCKbHRP6mYOkOQItI/gRKBBr0q0j9XmVfw6UzTPxOf2nfFh9U/elioYJzy1z9ldep+1MDZP//ZKAfojto/KFh1sxnF2j/mwWTxpwHdP+oKJiSUFd0/ZbHJyl2L3j/mITARU23gPzKmu8wNsOA/i/h/do5R4T+BAJFF5wviP/HZfTqz3uI/yzumsSRi4z/C/+lKBF/kP6XErxocveU/i3PFCaCt5j+/FInaV67mP6nOAndWCug/G1Tyzcip6D9sc2y1z7bqP8DOP+Qpte0/1z14rWVl7j9k/0KfV4vvPxbt+BJQpPE/O84SNIu68T/2PH1PsPPxP++LtNVSc/I/BNBOba8+8z8/F/2AI072Px9wzuwPHfc/l5gWaiyX+T+sRQdERJ/6P5LJs5eYDv8/9gg2BzdtAEA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"kFx1kXu45T8KBGvxJ/3lP0An6eztp+g/RuAh9U3W6j+Uv1ARnD3wPy1amQ8/1vI/4oAB5lVW8z/H54yXsXPzP8Fe4Het9fQ/oa9zMi379D/CBuwPcx31PypNa8AyhfU/qiyHOYuh9T8IGmBL3Hj3P74CNOZkmfc/PAJDPiy79z/vJwlAjTH4PwdFwmUq+/g/EARfrv5o+T9Kw5KNg4L5PwzmGXzam/k/2Nja6gkJ+j8tNJQ+wyf6Pz1MJPRqLvo/G2AFdrsx+j9hIj267j36P2pRwQs7U/o/QQGtRt/D+j+D6TXJDxX7P6BDWUcuFfs/THfGKjoi+z/HMaDVUzP7PzwtK1+ddfs/B8uTTNyE+z987jvvY6j7P0HZ2H7cKvw/idVhwi+A/D8+tt7im/z8PwrsLR3SAv0/nYiMfswx/T9/XPz4T139PzBQHABAiv0/QT9+Lhmv/T/kqWP2gzL+P4tyhPa0Uv4/7iE+hNZW/j9leWE+oM/+P3a07kqPS/8/AnXHRYtk/z/HrYpV55D/P3I1qEGYnP8/Bt6QhQAwAED08tlFaTcAQLokotNqhABAOMsEApWtAEBwDQJ65LYAQGYGsKxmuABAacgRUDbZAEBdyQs2i/4AQD+XpEoNCwFAVRmSMFAWAUBHbCCQEhsBQIupMwQpIgFAKBFo0KsiAUCVeQWfzjQBQPGpfVd8WAFAiIUKxil/AUBWp+5HDZwBQKCNcoDuqAFAglU3m1GsAUAeTBZ/GtABQK9gQkJZ0QFAFpus3LXoAUA9BCZiqg0CQMZ0l7kBFgJAEf/PzjEqAkAQILLofEECQD67T2fWWwJAecc0lkRsAkD4P12J4IsCQJX4VYOjtwJAca44AbTVAkCYIlH7ytUCQNVZ4M5KAQNAg0q+GTkVA0Bujq322VYDQNj5hzyltgNAuwevtazMA0DsX+jzavEDQEY7vgQUaQRAj7MEzaJuBEA+T98T7HwEQPwibbXUnARAAbRT26vPBEDQRT/giJMFQAicM/tDxwVAJqaFGstlBkBr0QER0acGQGTy7CWmwwdAewSbg5s2CEA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"69152"},"selection_policy":{"id":"69153"}},"id":"69139","type":"ColumnDataSource"},{"attributes":{},"id":"69146","type":"BasicTickFormatter"},{"attributes":{},"id":"69150","type":"Selection"},{"attributes":{},"id":"69153","type":"UnionRenderers"},{"attributes":{},"id":"69103","type":"LinearScale"},{"attributes":{"overlay":{"id":"69123"}},"id":"69117","type":"BoxZoomTool"},{"attributes":{"overlay":{"id":"69124"}},"id":"69119","type":"LassoSelectTool"},{"attributes":{"source":{"id":"69134"}},"id":"69138","type":"CDSView"},{"attributes":{},"id":"69108","type":"BasicTicker"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"69150"},"selection_policy":{"id":"69151"}},"id":"69134","type":"ColumnDataSource"},{"attributes":{},"id":"69118","type":"WheelZoomTool"},{"attributes":{"axis":{"id":"69111"},"dimension":1,"ticker":null},"id":"69114","type":"Grid"},{"attributes":{},"id":"69099","type":"DataRange1d"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"69136","type":"Patch"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"69123","type":"BoxAnnotation"},{"attributes":{"callback":null},"id":"69122","type":"HoverTool"},{"attributes":{},"id":"69112","type":"BasicTicker"},{"attributes":{},"id":"69152","type":"Selection"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"69115"},{"id":"69116"},{"id":"69117"},{"id":"69118"},{"id":"69119"},{"id":"69120"},{"id":"69121"},{"id":"69122"}]},"id":"69125","type":"Toolbar"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"69141","type":"Line"},{"attributes":{},"id":"69121","type":"SaveTool"},{"attributes":{},"id":"69116","type":"PanTool"},{"attributes":{},"id":"69151","type":"UnionRenderers"},{"attributes":{},"id":"69115","type":"ResetTool"},{"attributes":{},"id":"69120","type":"UndoTool"},{"attributes":{},"id":"69105","type":"LinearScale"},{"attributes":{"data_source":{"id":"69139"},"glyph":{"id":"69140"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"69141"},"selection_glyph":null,"view":{"id":"69143"}},"id":"69142","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"69107"},"ticker":null},"id":"69110","type":"Grid"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"69140","type":"Line"},{"attributes":{},"id":"69101","type":"DataRange1d"},{"attributes":{"text":""},"id":"69144","type":"Title"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"69124","type":"PolyAnnotation"},{"attributes":{"below":[{"id":"69107"}],"center":[{"id":"69110"},{"id":"69114"}],"left":[{"id":"69111"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"69137"},{"id":"69142"}],"title":{"id":"69144"},"toolbar":{"id":"69125"},"toolbar_location":"above","x_range":{"id":"69099"},"x_scale":{"id":"69103"},"y_range":{"id":"69101"},"y_scale":{"id":"69105"}},"id":"69098","subtype":"Figure","type":"Plot"},{"attributes":{"formatter":{"id":"69148"},"ticker":{"id":"69112"}},"id":"69111","type":"LinearAxis"},{"attributes":{"data_source":{"id":"69134"},"glyph":{"id":"69135"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"69136"},"selection_glyph":null,"view":{"id":"69138"}},"id":"69137","type":"GlyphRenderer"},{"attributes":{},"id":"69148","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"69139"}},"id":"69143","type":"CDSView"},{"attributes":{"formatter":{"id":"69146"},"ticker":{"id":"69108"}},"id":"69107","type":"LinearAxis"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"69135","type":"Patch"}],"root_ids":["69098"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"e8d63e5e-05a4-441a-841c-9be9b2db2e73","root_ids":["69098"],"roots":{"69098":"024ce446-433a-4b07-bad2-aaae5d675d6e"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();