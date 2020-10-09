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
    
      
      
    
      var element = document.getElementById("b64c78d5-2586-420c-917d-f5c5bb924c01");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'b64c78d5-2586-420c-917d-f5c5bb924c01' but no matching script tag was found.")
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
                    
                  var docs_json = '{"afd49080-5b6d-4e1c-8009-247807cc1466":{"roots":{"references":[{"attributes":{"below":[{"id":"68226"}],"center":[{"id":"68229"},{"id":"68233"}],"left":[{"id":"68230"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"68256"},{"id":"68258"}],"title":{"id":"68259"},"toolbar":{"id":"68244"},"toolbar_location":null,"x_range":{"id":"68218"},"x_scale":{"id":"68222"},"y_range":{"id":"68220"},"y_scale":{"id":"68224"}},"id":"68217","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"68264","type":"BasicTickFormatter"},{"attributes":{"toolbar":{"id":"68272"},"toolbar_location":"above"},"id":"68273","type":"ToolbarBox"},{"attributes":{"children":[[{"id":"68217"},0,0]]},"id":"68271","type":"GridBox"},{"attributes":{},"id":"68224","type":"LinearScale"},{"attributes":{"children":[{"id":"68273"},{"id":"68271"}]},"id":"68274","type":"Column"},{"attributes":{"callback":null},"id":"68241","type":"HoverTool"},{"attributes":{},"id":"68267","type":"UnionRenderers"},{"attributes":{"axis_label":"ESS for quantiles","formatter":{"id":"68264"},"ticker":{"id":"68231"}},"id":"68230","type":"LinearAxis"},{"attributes":{},"id":"68218","type":"DataRange1d"},{"attributes":{},"id":"68262","type":"BasicTickFormatter"},{"attributes":{},"id":"68227","type":"BasicTicker"},{"attributes":{"overlay":{"id":"68243"}},"id":"68238","type":"LassoSelectTool"},{"attributes":{"text":"sigma"},"id":"68259","type":"Title"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"68234"},{"id":"68235"},{"id":"68236"},{"id":"68237"},{"id":"68238"},{"id":"68239"},{"id":"68240"},{"id":"68241"}]},"id":"68244","type":"Toolbar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"68242","type":"BoxAnnotation"},{"attributes":{"axis":{"id":"68230"},"dimension":1,"ticker":null},"id":"68233","type":"Grid"},{"attributes":{},"id":"68237","type":"WheelZoomTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"68255","type":"Circle"},{"attributes":{"data_source":{"id":"68253"},"glyph":{"id":"68254"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68255"},"selection_glyph":null,"view":{"id":"68257"}},"id":"68256","type":"GlyphRenderer"},{"attributes":{},"id":"68231","type":"BasicTicker"},{"attributes":{},"id":"68235","type":"PanTool"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"68254","type":"Circle"},{"attributes":{"axis":{"id":"68226"},"ticker":null},"id":"68229","type":"Grid"},{"attributes":{},"id":"68239","type":"UndoTool"},{"attributes":{},"id":"68222","type":"LinearScale"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"R2riOHxslUDJrxt4rb2XQC8UfR7VtJlAMRHUGWndmUBKO3TVSyObQNp1b/0mJp1ASsbCePPwnkB+Du/cq5qgQH8ihBoHoKBAjJ8qLZB5oECYlOwhLnyfQD8CvMP22p1A58Gm42rqnEALInuU09KdQICWYY7d25xA2A/0ZSlsnEBzLBEly1mdQE4F40OedZlAKS7heDC7m0BmB8tcKnmYQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"68266"},"selection_policy":{"id":"68267"}},"id":"68253","type":"ColumnDataSource"},{"attributes":{},"id":"68220","type":"DataRange1d"},{"attributes":{},"id":"68234","type":"ResetTool"},{"attributes":{"source":{"id":"68253"}},"id":"68257","type":"CDSView"},{"attributes":{},"id":"68266","type":"Selection"},{"attributes":{},"id":"68240","type":"SaveTool"},{"attributes":{"overlay":{"id":"68242"}},"id":"68236","type":"BoxZoomTool"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"68262"},"ticker":{"id":"68227"}},"id":"68226","type":"LinearAxis"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"68243","type":"PolyAnnotation"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"68258","type":"Span"},{"attributes":{"toolbars":[{"id":"68244"}],"tools":[{"id":"68234"},{"id":"68235"},{"id":"68236"},{"id":"68237"},{"id":"68238"},{"id":"68239"},{"id":"68240"},{"id":"68241"}]},"id":"68272","type":"ProxyToolbar"}],"root_ids":["68274"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"afd49080-5b6d-4e1c-8009-247807cc1466","root_ids":["68274"],"roots":{"68274":"b64c78d5-2586-420c-917d-f5c5bb924c01"}}];
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