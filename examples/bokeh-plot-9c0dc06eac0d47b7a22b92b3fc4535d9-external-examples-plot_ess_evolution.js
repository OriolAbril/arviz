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
    
      
      
    
      var element = document.getElementById("6ab07b12-29f4-4cd7-9a40-b84cc9651771");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '6ab07b12-29f4-4cd7-9a40-b84cc9651771' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
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
                    
                  var docs_json = '{"cbabc40e-f83f-473a-b1a7-bbf69978684c":{"roots":{"references":[{"attributes":{"axis":{"id":"17261"},"ticker":null},"id":"17264","type":"Grid"},{"attributes":{"text":"b"},"id":"17312","type":"Title"},{"attributes":{"axis_label":"ESS","formatter":{"id":"17318"},"ticker":{"id":"17266"}},"id":"17265","type":"LinearAxis"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"17289","type":"Circle"},{"attributes":{"callback":null},"id":"17276","type":"HoverTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"orange"},"line_alpha":{"value":0.1},"line_color":{"value":"orange"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"17305","type":"Circle"},{"attributes":{},"id":"17255","type":"DataRange1d"},{"attributes":{"axis":{"id":"17265"},"dimension":1,"ticker":null},"id":"17268","type":"Grid"},{"attributes":{"data_source":{"id":"17293"},"glyph":{"id":"17294"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17295"},"selection_glyph":null,"view":{"id":"17297"}},"id":"17296","type":"GlyphRenderer"},{"attributes":{},"id":"17266","type":"BasicTicker"},{"attributes":{"label":{"value":"bulk"},"renderers":[{"id":"17291"},{"id":"17296"}]},"id":"17310","type":"LegendItem"},{"attributes":{"data_source":{"id":"17303"},"glyph":{"id":"17304"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17305"},"selection_glyph":null,"view":{"id":"17307"}},"id":"17306","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"17290","type":"Circle"},{"attributes":{},"id":"17327","type":"UnionRenderers"},{"attributes":{"overlay":{"id":"17277"}},"id":"17271","type":"BoxZoomTool"},{"attributes":{"source":{"id":"17303"}},"id":"17307","type":"CDSView"},{"attributes":{},"id":"17270","type":"PanTool"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"17322"},"selection_policy":{"id":"17321"}},"id":"17288","type":"ColumnDataSource"},{"attributes":{},"id":"17318","type":"BasicTickFormatter"},{"attributes":{},"id":"17269","type":"ResetTool"},{"attributes":{},"id":"17253","type":"DataRange1d"},{"attributes":{"click_policy":"hide","items":[{"id":"17310"},{"id":"17311"}],"location":"center_right","orientation":"horizontal"},"id":"17309","type":"Legend"},{"attributes":{},"id":"17275","type":"SaveTool"},{"attributes":{"label":{"value":"tail"},"renderers":[{"id":"17306"},{"id":"17301"}]},"id":"17311","type":"LegendItem"},{"attributes":{},"id":"17326","type":"Selection"},{"attributes":{},"id":"17272","type":"WheelZoomTool"},{"attributes":{"toolbar":{"id":"17331"},"toolbar_location":"above"},"id":"17332","type":"ToolbarBox"},{"attributes":{"above":[{"id":"17309"}],"below":[{"id":"17261"}],"center":[{"id":"17264"},{"id":"17268"}],"left":[{"id":"17265"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"17291"},{"id":"17296"},{"id":"17301"},{"id":"17306"},{"id":"17308"}],"title":{"id":"17312"},"toolbar":{"id":"17279"},"toolbar_location":null,"x_range":{"id":"17253"},"x_scale":{"id":"17257"},"y_range":{"id":"17255"},"y_scale":{"id":"17259"}},"id":"17252","subtype":"Figure","type":"Plot"},{"attributes":{"overlay":{"id":"17278"}},"id":"17273","type":"LassoSelectTool"},{"attributes":{},"id":"17274","type":"UndoTool"},{"attributes":{"source":{"id":"17298"}},"id":"17302","type":"CDSView"},{"attributes":{},"id":"17321","type":"UnionRenderers"},{"attributes":{},"id":"17328","type":"Selection"},{"attributes":{"line_alpha":0.1,"line_color":"orange","x":{"field":"x"},"y":{"field":"y"}},"id":"17300","type":"Line"},{"attributes":{"toolbars":[{"id":"17279"}],"tools":[{"id":"17269"},{"id":"17270"},{"id":"17271"},{"id":"17272"},{"id":"17273"},{"id":"17274"},{"id":"17275"},{"id":"17276"}]},"id":"17331","type":"ProxyToolbar"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"17269"},{"id":"17270"},{"id":"17271"},{"id":"17272"},{"id":"17273"},{"id":"17274"},{"id":"17275"},{"id":"17276"}]},"id":"17279","type":"Toolbar"},{"attributes":{},"id":"17322","type":"Selection"},{"attributes":{},"id":"17323","type":"UnionRenderers"},{"attributes":{"line_color":"#1f77b4","x":{"field":"x"},"y":{"field":"y"}},"id":"17294","type":"Line"},{"attributes":{"data_source":{"id":"17288"},"glyph":{"id":"17289"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17290"},"selection_glyph":null,"view":{"id":"17292"}},"id":"17291","type":"GlyphRenderer"},{"attributes":{},"id":"17324","type":"Selection"},{"attributes":{},"id":"17316","type":"BasicTickFormatter"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"17324"},"selection_policy":{"id":"17323"}},"id":"17293","type":"ColumnDataSource"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"17277","type":"BoxAnnotation"},{"attributes":{"children":[[{"id":"17252"},0,0]]},"id":"17330","type":"GridBox"},{"attributes":{"source":{"id":"17288"}},"id":"17292","type":"CDSView"},{"attributes":{"source":{"id":"17293"}},"id":"17297","type":"CDSView"},{"attributes":{"fill_color":{"value":"orange"},"line_color":{"value":"orange"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"17304","type":"Circle"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"17328"},"selection_policy":{"id":"17327"}},"id":"17303","type":"ColumnDataSource"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"17308","type":"Span"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"17278","type":"PolyAnnotation"},{"attributes":{},"id":"17257","type":"LinearScale"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","x":{"field":"x"},"y":{"field":"y"}},"id":"17295","type":"Line"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"17326"},"selection_policy":{"id":"17325"}},"id":"17298","type":"ColumnDataSource"},{"attributes":{},"id":"17325","type":"UnionRenderers"},{"attributes":{"axis_label":"Total number of draws","formatter":{"id":"17316"},"ticker":{"id":"17262"}},"id":"17261","type":"LinearAxis"},{"attributes":{"line_color":"orange","x":{"field":"x"},"y":{"field":"y"}},"id":"17299","type":"Line"},{"attributes":{"data_source":{"id":"17298"},"glyph":{"id":"17299"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17300"},"selection_glyph":null,"view":{"id":"17302"}},"id":"17301","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"17332"},{"id":"17330"}]},"id":"17333","type":"Column"},{"attributes":{},"id":"17262","type":"BasicTicker"},{"attributes":{},"id":"17259","type":"LinearScale"}],"root_ids":["17333"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"cbabc40e-f83f-473a-b1a7-bbf69978684c","root_ids":["17333"],"roots":{"17333":"6ab07b12-29f4-4cd7-9a40-b84cc9651771"}}];
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