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
    
      
      
    
      var element = document.getElementById("8799b0a9-27a4-491d-8f7e-5c0d38d11a1a");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '8799b0a9-27a4-491d-8f7e-5c0d38d11a1a' but no matching script tag was found.")
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
                    
                  var docs_json = '{"3fd8d5fa-e813-48ba-b90d-09c1821f8589":{"roots":{"references":[{"attributes":{},"id":"5333","type":"LinearScale"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5345"},{"id":"5346"},{"id":"5347"},{"id":"5348"},{"id":"5349"},{"id":"5350"},{"id":"5351"},{"id":"5352"}]},"id":"5355","type":"Toolbar"},{"attributes":{"overlay":{"id":"5354"}},"id":"5349","type":"LassoSelectTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5354","type":"PolyAnnotation"},{"attributes":{},"id":"5342","type":"BasicTicker"},{"attributes":{"data":{"x":{"__ndarray__":"szilcaCrCcDpDUI7B9kAwI6cj/D9NQDAdAkm2HJU/r8jI5h6h9r9v9TYqf8MBPm//bHOtwS++L/OULTVv/f2v/0jYTQftPW/AsIjC3n19L/0HibHhSz0vxrseTGq4vO/YwQUSBME8r/skyj/Terxv9qcYw4kGfG/+fzxJkwO8b+hvGBGZNnwvz5hCPJGVu6/ZQFJFvDZ7L85cHdLmorsv9Qio3vBZeu/kuEcivPK6b8068YAG6jpv359ixqs3Oe/AuyEExHH57897eElE/nmv24YmCkSYea/TEy8IxC95b8+ohtBRCnjv38dg8lcz+C/iAQUXqwj4L9alWr0h87fv/rfUpyXqt+/7qYk585C3L+9lwve/inav06DPaj2KNm/bRBes0Du1b+mPco/sqPUv2vZDFsyxtK/nRrdZNDUz7/rmhGJJnvNvzhg0DJEW8W/6eSDqLcBxb+TvcnHGcrBv9bTwfbeVMG/rg0vC6apwL8gUeWvhVy8vxZWgb2T67C/HIRha3bTqr8CgTJaCLCjv96zMR/x4Xy/OgRrsH5YZL8YoAmNT2RSP18inJg01Hw/tKoFcczkhD91zFDWfuDCP2dOB8eiTMU/sgaA6vB7xT/GzQjuCVzGPzZhPdfaj8c/oM02E/RfyD+gk29mEWLJPwJfbBObSMo/rqPqhenlyz9Y14zgYoXPP2qY7cXWLNI/6vRR/NK/2j+/hikztMnaP1ntJ6R57Ns/z9SMQG6d3j+WcOqYMSXgP9jukVANTOA/GKP8fNpM4D/wHT+Ne6nhPyAv9zFlB+I/okwRxMAo4j87Mx5cN5rmP+euG8hN2uY/PzMQgz0s5z9GEK8v7aPnPwL7fQz/Juk/SQa7X5xE6z/xAZNkTCjsPypYa7+94+4/oLGlp3u87z+DkPSHOfHvPwuN+pRTCfA/UtXQeqhX8D9oLj3kMeLwPwDqhHyAdfI/o2A8N1SI8j8fskfiuVjzP03UTeSNbfY/JqD5LDGV9z953rtM7Bb5P4MNZYRYMPo/+r8V8g6s+j9PgKG6eMv9P5T6Ru0DJAZApIk3qEFJBkA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"NB1rOX5R2T8u5HuJ8U3uP+TG4B4ElO8/Rvvsk8bV8D9u7rNCvBLxP5YTK4D5ffM/AqcYpP2g8z+Z1yUVIIT0PwJuz2XwJfU//x5uekOF9T+G8Gwcven1P/MJQ+eqDvY/zv31W/b99j8KtmsA2Qr3P5Mxzvhtc/c/hAGH7Nl49z+woc/cTZP3P7DnfUNuavg/p79t+oPJ+D/yIyJtWd34P0s3F6GPJvk/nMd4HUON+T8zRc4/+ZX5P6AgXfnUCPo/AMUeuzsO+j+xhIc2u0H6P+T5mXW7Z/o/7ewQ97uQ+j9wF7nvrjX7P6A4n80ozPs/3v566BT3+z9VrXIBLwb8PwGkdQytCvw/ImsbI6Z3/D8IjT4kwLr8P5ZP+Crh2vw/8j2U6TdC/T9LuAa4iWv9P9NknrQ5p/0/Vi6y+bIC/j9R5m6XTSj+P/z50rxLqv4/scF3heSv/j8nZINjXuP+P8PikxCy6v4/JQ9Nn2X1/j931YDSGx3/P0/1E2KjeP8/8HlSJrKU/z/8NZfeP7H/PyZncAeP8f8/P+VT4On6/z+a0PhEJgEAQAknJg01BwBA1YI4ZnIKAEBkhrL2A5cAQHM6OBZlqgBANgBUh9+rAEBuRnBP4LIAQArrudZ+vABAbbaZoP/CAECdfDOLEMsAQPhim9hE0gBAHVUvTC/fAEC7ZgQXK/wAQIfZXmzNIgFATx/FL/2rAUBsmDJDm6wBQNZ+QprHvgFATc0I5NbpAUATTh0zpgQCQNs9EqqBCQJAY5SfT5sJAkC+46dxLzUCQOTlPqbsQAJAlCmCGBhFAkBnxoPrRtMCQN11A7lJ2wJAaAZisIflAkAJ4vWlffQCQGC/j+HfJANAyWD3i5NoA0A+YJKMCYUDQAVr7bd33ANANLb0dI/3A0AQkv4wJ/4DQEOjPuVUAgRAVDW0HuoVBECaSw95jDgEQIA6IR9gnQRAKRjPDRWiBECI7JF4LtYEQBN1E3ljmwVACmg+S0zlBUCe9y4Tu0UGQGFDGSEWjAZA/m+FvAOrBkAUYKgu3nIHQEp9o/YBEgtA0sQb1KAkC0A=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5384"},"selection_policy":{"id":"5385"}},"id":"5369","type":"ColumnDataSource"},{"attributes":{"text":""},"id":"5374","type":"Title"},{"attributes":{},"id":"5329","type":"DataRange1d"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5382"},"selection_policy":{"id":"5383"}},"id":"5364","type":"ColumnDataSource"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5366","type":"Patch"},{"attributes":{},"id":"5346","type":"PanTool"},{"attributes":{},"id":"5345","type":"ResetTool"},{"attributes":{"callback":null},"id":"5352","type":"HoverTool"},{"attributes":{},"id":"5385","type":"UnionRenderers"},{"attributes":{},"id":"5331","type":"DataRange1d"},{"attributes":{"data_source":{"id":"5364"},"glyph":{"id":"5365"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5366"},"selection_glyph":null,"view":{"id":"5368"}},"id":"5367","type":"GlyphRenderer"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5365","type":"Patch"},{"attributes":{},"id":"5383","type":"UnionRenderers"},{"attributes":{},"id":"5382","type":"Selection"},{"attributes":{},"id":"5350","type":"UndoTool"},{"attributes":{"source":{"id":"5369"}},"id":"5373","type":"CDSView"},{"attributes":{"below":[{"id":"5337"}],"center":[{"id":"5340"},{"id":"5344"}],"left":[{"id":"5341"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5367"},{"id":"5372"}],"title":{"id":"5374"},"toolbar":{"id":"5355"},"toolbar_location":"above","x_range":{"id":"5329"},"x_scale":{"id":"5333"},"y_range":{"id":"5331"},"y_scale":{"id":"5335"}},"id":"5328","subtype":"Figure","type":"Plot"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5353","type":"BoxAnnotation"},{"attributes":{},"id":"5348","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"5369"},"glyph":{"id":"5370"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5371"},"selection_glyph":null,"view":{"id":"5373"}},"id":"5372","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"5341"},"dimension":1,"ticker":null},"id":"5344","type":"Grid"},{"attributes":{},"id":"5377","type":"BasicTickFormatter"},{"attributes":{},"id":"5338","type":"BasicTicker"},{"attributes":{},"id":"5335","type":"LinearScale"},{"attributes":{"axis":{"id":"5337"},"ticker":null},"id":"5340","type":"Grid"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5371","type":"Line"},{"attributes":{"formatter":{"id":"5377"},"ticker":{"id":"5342"}},"id":"5341","type":"LinearAxis"},{"attributes":{"overlay":{"id":"5353"}},"id":"5347","type":"BoxZoomTool"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5370","type":"Line"},{"attributes":{"source":{"id":"5364"}},"id":"5368","type":"CDSView"},{"attributes":{},"id":"5384","type":"Selection"},{"attributes":{},"id":"5379","type":"BasicTickFormatter"},{"attributes":{},"id":"5351","type":"SaveTool"},{"attributes":{"formatter":{"id":"5379"},"ticker":{"id":"5338"}},"id":"5337","type":"LinearAxis"}],"root_ids":["5328"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"3fd8d5fa-e813-48ba-b90d-09c1821f8589","root_ids":["5328"],"roots":{"5328":"8799b0a9-27a4-491d-8f7e-5c0d38d11a1a"}}];
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